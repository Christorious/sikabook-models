"""In-memory SQLite sandbox mirroring the on-device Room schema. One per episode."""
import sqlite3
from dataclasses import dataclass
from typing import Optional
from .schema import ActionType, ModelActionOutput, SCREEN_ACTION_MASK


@dataclass
class GateConfig:
    # Placeholders. Tune on real enrolled-vendor data (EER point), don't trust these.
    speaker_threshold: float = 0.68
    # Below speaker_threshold but at or above this: the utterance is NOT discarded, it is held
    # for the day-end review she already sits through. The ledger is unchanged either way, but
    # a held utterance is recoverable and a rejected one is gone for good. None keeps the old
    # binary gate, so every number measured before this existed still reproduces.
    review_threshold: Optional[float] = None
    min_confidence: float = 0.55
    max_amount_ghs: float = 20_000.0   # above this, ask instead of writing: catches unconverted old cedis
    official_momo_senders: tuple = ("MobileMoney", "MoMo", "TelecelCash", "AT Money")


class LocalLedgerSandbox:
    def __init__(self, momo_inbox=None, credit_book=None, cfg: GateConfig = None):
        self.cfg = cfg or GateConfig()
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
            CREATE TABLE transactions (id INTEGER PRIMARY KEY, tx_type TEXT,
                item TEXT, quantity INTEGER, amount_ghs REAL);
            CREATE TABLE credit_book (id INTEGER PRIMARY KEY, customer TEXT,
                item TEXT, amount_owed REAL, status TEXT);
        """)
        for row in credit_book or []:
            self.conn.execute("INSERT INTO credit_book (customer,item,amount_owed,status) VALUES (?,?,?,?)",
                              (row["customer"], row["item"], row["amount_owed"], row["status"]))
        self.momo_inbox = momo_inbox or []
        self.flags = {"blocked_speaker": False, "clarification_requested": False,
                      "scam_alert": False, "illegal_action": False,
                      "deferred_for_review": False}

    def step(self, p: ModelActionOutput):
        c = self.cfg
        if p.speaker_similarity < c.speaker_threshold:          # 1. voice gate: accept / review / reject
            if c.review_threshold is not None and p.speaker_similarity >= c.review_threshold:
                self.flags["deferred_for_review"] = True
            else:
                self.flags["blocked_speaker"] = True
            return
        if p.action not in SCREEN_ACTION_MASK[p.screen]:         # 2. screen mask
            self.flags["illegal_action"] = True; return
        if p.action == ActionType.ASK_CLARIFICATION or p.confidence < c.min_confidence:
            self.flags["clarification_requested"] = True; return  # 3. never guess money
        if p.amount_ghs > c.max_amount_ghs:                     # 4. implausible amount, e.g. 200000 not 20
            self.flags["clarification_requested"] = True; return
        x = self.conn.execute
        if p.action == ActionType.LOG_CASH_SALE:
            x("INSERT INTO transactions (tx_type,item,quantity,amount_ghs) VALUES ('CASH_SALE',?,?,?)",
              (p.item, p.quantity, p.amount_ghs))
        elif p.action == ActionType.LOG_EXPENSE:
            x("INSERT INTO transactions (tx_type,item,quantity,amount_ghs) VALUES ('EXPENSE',?,?,?)",
              (p.item, p.quantity, p.amount_ghs))
        elif p.action == ActionType.LOG_CREDIT_SALE:
            x("INSERT INTO credit_book (customer,item,amount_owed,status) VALUES (?,?,?,'UNPAID')",
              (p.customer_name, p.item, p.amount_ghs))
        elif p.action == ActionType.LOG_DEBT_PAYMENT:
            row = x("SELECT id, amount_owed FROM credit_book WHERE customer=? AND status='UNPAID'",
                    (p.customer_name,)).fetchone()
            if row:  # partial payments reduce the debt; full payment closes it
                left = round(row["amount_owed"] - p.amount_ghs, 2)
                x("UPDATE credit_book SET amount_owed=?, status=? WHERE id=?",
                  (max(left, 0.0), "PAID" if left <= 0 else "UNPAID", row["id"]))
        elif p.action == ActionType.VERIFY_MOMO_SMS:
            ok = any(m["sender"] in c.official_momo_senders and abs(m["amount"] - p.amount_ghs) < 0.01
                     for m in self.momo_inbox)
            if ok:
                x("INSERT INTO transactions (tx_type,item,quantity,amount_ghs) VALUES ('MOMO_SALE',?,?,?)",
                  (p.item, p.quantity, p.amount_ghs))
            else:
                self.flags["scam_alert"] = True

    def snapshot(self):
        q = lambda s: [dict(r) for r in self.conn.execute(s)]
        return {"transactions": q("SELECT tx_type,item,quantity,amount_ghs FROM transactions ORDER BY id"),
                "credit_book": q("SELECT customer,item,amount_owed,status FROM credit_book ORDER BY id"),
                **self.flags}
