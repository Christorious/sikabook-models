"""Bounded, screen-constrained action space ("Thud" style). Stdlib only."""
from dataclasses import dataclass, asdict
from enum import IntEnum
from typing import Optional


class ScreenState(IntEnum):
    HOME_LEDGER = 0
    CONFIRM_MODAL = 1
    CREDIT_BOOK = 2
    MOMO_VERIFY = 3


class ActionType(IntEnum):
    NO_OP = 0
    LOG_CASH_SALE = 1
    LOG_CREDIT_SALE = 2
    LOG_DEBT_PAYMENT = 3
    LOG_EXPENSE = 4
    VERIFY_MOMO_SMS = 5
    CONFIRM_ACTION = 6
    CANCEL_ACTION = 7
    ASK_CLARIFICATION = 8


# Which actions are even legal on each screen. The model head is masked to these.
SCREEN_ACTION_MASK = {
    ScreenState.HOME_LEDGER: {ActionType.LOG_CASH_SALE, ActionType.LOG_CREDIT_SALE,
                              ActionType.LOG_EXPENSE, ActionType.VERIFY_MOMO_SMS,
                              ActionType.ASK_CLARIFICATION, ActionType.NO_OP},
    ScreenState.CONFIRM_MODAL: {ActionType.CONFIRM_ACTION, ActionType.CANCEL_ACTION,
                                ActionType.ASK_CLARIFICATION},
    ScreenState.CREDIT_BOOK: {ActionType.LOG_CREDIT_SALE, ActionType.LOG_DEBT_PAYMENT,
                              ActionType.ASK_CLARIFICATION, ActionType.NO_OP},
    ScreenState.MOMO_VERIFY: {ActionType.VERIFY_MOMO_SMS, ActionType.CONFIRM_ACTION,
                              ActionType.CANCEL_ACTION, ActionType.ASK_CLARIFICATION},
}

MARKET_ITEMS = ["NONE", "tilapia", "kpanla", "salmon", "mackerel", "red_fish",
                "herrings", "catfish", "shrimp", "ice_block", "charcoal"]

# Reference only (for synthetic data + slot tagging). Verify spellings with native speakers.
# Note the trap pair: aduonu = 20, aduonum = 50.
TWI_NUMBERS = {"baako": 1, "mmienu": 2, "mmiɛnsa": 3, "ɛnan": 4, "enum": 5, "nsia": 6,
               "nson": 7, "nwɔtwe": 8, "nkron": 9, "du": 10, "aduonu": 20,
               "aduasa": 30, "aduanan": 40, "aduonum": 50, "aduosia": 60,
               "aduɔson": 70, "aduowɔtwe": 80, "aduokron": 90, "ɔha": 100, "ahanu": 200}

# Old cedis: many traders still quote prices in pre-2007 cedis ("seven million" = GH₵700).
# 10,000 old cedis = GH₵1. Twi: apem = 1,000, mpem = thousands, ɔpepem = 1,000,000 (verify spellings).
OLD_CEDI_PER_GHS = 10_000


def old_cedi_to_ghs(old):
    return round(old / OLD_CEDI_PER_GHS, 2)


@dataclass
class ModelActionOutput:
    """What sikabook-models emits for one utterance."""
    speaker_similarity: float          # cosine sim to enrolled vendor, 0..1
    action: ActionType
    item: str = "NONE"
    quantity: int = 0
    amount_ghs: float = 0.0
    customer_name: Optional[str] = None
    confidence: float = 1.0            # money/slot confidence; low => clarify, never guess
    screen: ScreenState = ScreenState.HOME_LEDGER

    def to_dict(self):
        d = asdict(self)
        d["action"] = self.action.name
        d["screen"] = self.screen.name
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["action"] = ActionType[d["action"]]
        d["screen"] = ScreenState[d.get("screen", "HOME_LEDGER")]
        return ModelActionOutput(**d)
