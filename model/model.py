from dataclasses import dataclass
from enum import Enum
from typing import List, Type, Any, Optional


class ConversationStage(Enum):
    """
    Captures a general stage of the conversation.
    """
    INFO_GROUP = "info_group"
    TIME_SELECTION = "time_selection"


class InfoGroupStage(Enum):
    """
    Captures a general location within an InformationGroup.
    """
    INITIAL_GROUP = "initial_group"
    MAIN_INFO = "main_info"


@dataclass
class InformationElement:
    """
    Holds information for a single unit/field of information from the patient.
    """
    element_name: str
    element_type: Type
    element_value: Optional[Any] = None
    topic: Optional[str] = None
    attempt: int = 0
    max_attempts: int = 3
    get_from_user: bool = True
    retrieval_query: Optional[str] = None


@dataclass
class InformationGroup:
    """
    Holds information for a group of related InformationElements.
    """
    info_elements: List[InformationElement]
    preamble: Optional[str] = None
    stage: InfoGroupStage = InfoGroupStage.INITIAL_GROUP
    current_element: int = 0


@dataclass
class AppointmentInfo:
    """
    Holds information related to the medical appointment.
    """
    date_and_time: Optional[str] = None
    doctor_name: Optional[str] = None
    doctor_availability_response: Optional[str] = None
    appointment_confirmed: bool = False
    confirmation_attempts = 0
    max_attempts = 2

