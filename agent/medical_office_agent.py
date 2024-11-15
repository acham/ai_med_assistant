from __future__ import print_function

import json
import logging
from datetime import datetime
from typing import Optional, AsyncGenerator, Tuple
from collections import OrderedDict
import os

from vocode.streaming.agent.base_agent import RespondAgent, BaseAgent, AgentResponseMessage, \
    tracer, AGENT_TRACE_NAME
from vocode.streaming.transcriber.base_transcriber import Transcription
from vocode.streaming.agent.factory import AgentFactory
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.message import BaseMessage
import openai

from model.model import InformationGroup, InformationElement, ConversationStage, InfoGroupStage, AppointmentInfo

_CHAT_MODEL = "gpt-3.5-turbo"

_CHAT_AGENT_INITIAL_MESSAGES = [
    {"role": "system",
     "content": "You are a helpful assistant whose task is to collect information from a medical patient."},
    {"role": "user",
     "content": "I am a medical patient. You are an office assistant."},
]


class MedicalOfficeAgentConfig(AgentConfig):
    info_groups: OrderedDict


def get_initial_agent_config():
    return (
        MedicalOfficeAgentConfig(
            allow_agent_to_be_cut_off=False,
            generate_responses=True,
            info_groups=OrderedDict(
                basic=InformationGroup(
                    preamble="I need to start by getting some basic information.",
                    info_elements=[InformationElement("name", str, topic="collect my name"),
                                   InformationElement("date of birth", datetime, topic="collect my date of birth")],
                    stage=InfoGroupStage.INITIAL_GROUP
                ),
                insurance=InformationGroup(
                    preamble="Now I need some insurance information.",
                    info_elements=[InformationElement("provider", str, topic="collect my insurance provider"),
                                   InformationElement("insurance ID", str, topic="collect my insurance ID number")],
                    stage=InfoGroupStage.MAIN_INFO
                ),
                referral=InformationGroup(
                    preamble=None,
                    info_elements=[
                        InformationElement("referral", bool, topic="determine whether I have a referral to a physician",
                                           retrieval_query="Do I have a referral"),
                        InformationElement("referred doctor", Optional[str],
                                           topic="get the name of the referred physician")],
                    stage=InfoGroupStage.MAIN_INFO
                ),
                reason_for_visit=InformationGroup(
                    preamble=None,
                    info_elements=[
                        InformationElement("visit_reason", str, topic="determine the main reason for my visit")],
                    stage=InfoGroupStage.MAIN_INFO
                ),
                contact_information=InformationGroup(
                    preamble="I'd like to get some additional contact information.",
                    info_elements=[InformationElement("address", str, topic="collect my address"),
                                   InformationElement("phone number", str, topic="collect my phone number")],
                    stage=InfoGroupStage.MAIN_INFO
                ),
            ),
            end_conversation_on_goodbye=False,
            initial_message=BaseMessage(text="Hello, this is the medical office."),
        ))


class MedicalOfficeAgent(RespondAgent[MedicalOfficeAgentConfig]):
    """
    Implements an agent responsible for collecting medical information from a patient and making appointments.
    """
    def __init__(
            self,
            agent_config: MedicalOfficeAgentConfig,
            logger: Optional[logging.Logger] = None,
    ):
        super().__init__(agent_config=agent_config, logger=logger)
        openai.api_key = os.environ.get("OPENAI_API_KEY")

        self.info_group_names = list(agent_config.info_groups.keys())
        self.info_group_index = 0
        self.conversation_stage = ConversationStage.INFO_GROUP

        self.conversation_id = None

        self.messages = _CHAT_AGENT_INITIAL_MESSAGES

        self.appointment_info = AppointmentInfo()
        self.contact_phone_number = ""

        self.end_conversation = False

    def init_connection(self, conversation_id: Optional[str] = None):
        self.agent_config = get_initial_agent_config()
        self.info_group_names = list(self.agent_config.info_groups.keys())
        self.info_group_index = 0
        self.conversation_stage = ConversationStage.INFO_GROUP

        self.conversation_id = conversation_id

        self.messages = _CHAT_AGENT_INITIAL_MESSAGES

        self.appointment_info = AppointmentInfo()
        self.contact_phone_number = ""

        self.end_conversation = False

    async def handle_generate_response(
            self, transcription: Transcription, conversation_id: str
    ) -> bool:
        """
        Overrides implementation from Vocode source to catch the 'should_stop' flag from
        the called function generate_response().
        :param transcription:
        :param conversation_id:
        :return: Whether to stop the conversation.
        """
        agent_span = tracer.start_span(
            AGENT_TRACE_NAME, {"generate_response": True}  # type: ignore
        )
        responses = self.generate_response(
            transcription.message,
            is_interrupt=transcription.is_interrupt,
            conversation_id=conversation_id,
        )
        is_first_response = True
        should_stop = False
        async for response in responses:
            self.logger.debug("response: %s", response[0])

            if is_first_response:
                agent_span.end()
            self.produce_interruptible_event_nonblocking(
                AgentResponseMessage(message=BaseMessage(text=response[0])),
                is_interruptible=self.agent_config.allow_agent_to_be_cut_off,
            )

            if response[1]:
                should_stop = True

        if should_stop:
            return True

        return False

    async def generate_response(
            self, human_input, is_interrupt: bool = False, conversation_id: Optional[str] = None
    ) -> AsyncGenerator[Tuple[str, bool], None]:  # message and whether or not the message is interruptible
        """
        The main conversation flow for the medical office agent.

        Iterates through the InformationGroup objects from the AGENT_CONFIG, reads
        appointment availability for doctors from an external data file, and attempts to confirm an appointment.

        :return: a generator that yields the agent's response one sentence at a time, and whether or not to
        end the conversation after the response.
        """
        self.log_state()

        if conversation_id != self.conversation_id:
            self.init_connection(conversation_id)

        voice_response = ""

        if self.end_conversation:
            yield "Goodbye", True

        if self.conversation_stage == ConversationStage.INFO_GROUP:
            info_group_key = self.info_group_names[self.info_group_index]
            info_group = self.agent_config.info_groups[info_group_key]

            if info_group.stage == InfoGroupStage.INITIAL_GROUP:
                # can ignore the user's input
                voice_response += info_group.preamble or ""
                voice_response += " " + self.get_query_for_element(info_group.info_elements[info_group.current_element])
                info_group.stage = InfoGroupStage.MAIN_INFO

            elif info_group.stage == InfoGroupStage.MAIN_INFO:
                self.add_message_to_chat(human_input, "user")

                if self.validate_user_input(info_group.info_elements[info_group.current_element]):
                    self.logger.debug("valid human_input: %s", human_input)

                    self.process_user_input(info_group)

                    info_group.current_element += 1
                    while info_group.current_element < len(info_group.info_elements) and \
                            not info_group.info_elements[info_group.current_element].get_from_user:
                        info_group.current_element += 1

                    if info_group.current_element >= len(info_group.info_elements):
                        self.info_group_index += 1

                    if self.info_group_index >= len(self.info_group_names):
                        self.conversation_stage = ConversationStage.TIME_SELECTION
                        voice_response += self.get_provider_times()
                    else:
                        # next info element, possibly in new group
                        info_group_key = self.info_group_names[self.info_group_index]
                        info_group = self.agent_config.info_groups[info_group_key]

                        if info_group.current_element == 0:
                            voice_response += info_group.preamble or ""

                        voice_response += " " + self.get_query_for_element(
                            info_group.info_elements[info_group.current_element])
                else:
                    # user input invalid
                    element = info_group.info_elements[info_group.current_element]
                    if element.attempt >= element.max_attempts:
                        yield "Goodbye", True
                    else:
                        voice_response += "Sorry, didn't get that."
                        voice_response += " " + self.get_query_for_element(
                            info_group.info_elements[info_group.current_element])
                        element.attempt += 1

        elif self.conversation_stage == ConversationStage.TIME_SELECTION:
            self.add_message_to_chat(human_input, "user")

            if self.check_appointment_confirmation():
                confirmation_response = f"Confirmed with {self.appointment_info.doctor_name} on {self.appointment_info.date_and_time}"
                self.end_conversation = True

                yield confirmation_response, False
            else:
                self.appointment_info.confirmation_attempts += 1
                if self.appointment_info.confirmation_attempts > self.appointment_info.max_attempts:
                    yield "Sorry, couldn't confirm appointment, goodbye", True
                else:
                    retry_response = "Sorry, let's try again. " + self.appointment_info.doctor_availability_response
                    self.add_message_to_chat(human_input, "assistant")
                    yield retry_response, False

        self.add_message_to_chat(voice_response, "assistant")

        yield voice_response, False

    def validate_user_input(self, info_element: InformationElement):
        """
        Checks with a chat model whether the information provided by the user is understandable and makes sense
        based on what was asked.

        :param info_element:
        :return: whether the information provided by the user is valid
        """
        chat_user_message = "Were you able to " + info_element.topic + " from my response, yes or no?"
        self.add_message_to_chat(chat_user_message, "user")

        self.logger.debug("validation chat user message: %s", chat_user_message)

        response = openai.ChatCompletion.create(
            model=_CHAT_MODEL,
            messages=self.messages,
            max_tokens=2
        )

        short_response = response.choices[0].message.content.strip()
        self.add_message_to_chat(short_response, "assistant")

        self.logger.debug("validation response: %s", short_response)

        if "yes" in short_response.lower():
            return True

        return False

    def process_user_input(self, info_group: InformationGroup):
        """
        For a single information element, attempts to retrieve the information as it was understood
        by the chat model, and attempts to store it in a field with an expected type.

        Can make special adjustments to other fields based on the value retrieve; here, no referral removes
        the request for a referred physician.

        :param info_group:
        :return: None
        """
        element = info_group.info_elements[info_group.current_element]

        if element.element_type == datetime:
            processed_date_components = []
            for date_component in ("year", "month", "day"):
                user_message = "Provide just the " + date_component + " for that date, as a number, without extra tokens."
                self.add_message_to_chat(user_message, "user")

                response = openai.ChatCompletion.create(
                    model=_CHAT_MODEL,
                    messages=self.messages,
                    temperature=0
                )

                response = response.choices[0].message.content.strip().strip(".")
                self.add_message_to_chat(response, "assistant")

                processed_date_components.append(int(response))

            element.element_value = datetime(processed_date_components[0],
                                             processed_date_components[1],
                                             processed_date_components[2])

        elif element.element_type == bool:
            user_message = element.retrieval_query + ", yes or no, no extra tokens?"
            self.add_message_to_chat(user_message, "user")

            response = openai.ChatCompletion.create(
                model=_CHAT_MODEL,
                messages=self.messages,
                max_tokens=1
            )

            response = response.choices[0].message.content.strip().strip(".")
            self.add_message_to_chat(response, "assistant")

            if "yes" in response.lower():
                element.element_value = True
            else:
                element.element_value = False

        else:
            chat_user_message = "Provide the " + element.element_name + " only, without extra tokens."
            self.add_message_to_chat(chat_user_message, "user")

            response = openai.ChatCompletion.create(
                model=_CHAT_MODEL,
                messages=self.messages)

            short_response = response.choices[0].message.content.strip().strip(".")
            self.add_message_to_chat(short_response, "assistant")

            self.logger.debug("process input, %s", short_response)
            element.element_value = short_response

            if element.element_name == "phone number":
                self.contact_phone_number = element.element_value

        # Only request a referral physician if there is a referral
        if element.element_name == "referral" and not element.element_value:
            for iter_element in info_group.info_elements:
                if iter_element.element_name == "referred doctor":
                    iter_element.get_from_user = False

    def get_query_for_element(self, info_element: InformationElement) -> str:
        """
        Uses the chat model to obtain a query for the user to collect a specific InformationElement.

        :param info_element:
        :return: The query as it will be asked to the user.
        """
        chat_user_message = "Generate a basic sentence to " + info_element.topic + "."

        self.add_message_to_chat(chat_user_message, "user")

        response = openai.ChatCompletion.create(
            model=_CHAT_MODEL,
            messages=self.messages)

        query_for_group = response.choices[0].message.content.strip()
        self.add_message_to_chat(query_for_group, "assistant")

        self.logger.debug("query: %s", query_for_group)

        return query_for_group

    def get_provider_times(self) -> str:
        """
        Gets available provider times, based on whether a referred physician was provided.
        Also informs the chat model so that it can help in scheduling an appointment later.
        :return: available provider times
        """
        with open("data/doctor_availability.json") as json_file:
            doctor_data = json.load(json_file)

        referral_info_group = self.agent_config.info_groups["referral"] if "referral" in self.agent_config.info_groups else None
        referral_exists = False
        referral_name = None

        if referral_info_group:
            for info_element in referral_info_group.info_elements:
                if info_element.element_name == "referral":
                    referral_exists = info_element.element_value
                if info_element.element_name == "referred doctor":
                    referral_name = info_element.element_value

        if referral_exists:
            doctor_name = referral_name
            times = doctor_data["New Doctor"]
        else:
            doctor_name = "Doctor Nirvana"
            times = doctor_data[doctor_name]

        available_times = " and ".join(times)
        response = f"Available times for {doctor_name} are {available_times}. Which time works best?"

        self.add_message_to_chat(response, "assistant")
        self.appointment_info.doctor_availability_response = response
        self.appointment_info.doctor_name = doctor_name

        return response

    def add_message_to_chat(self, message: str, role: str) -> None:
        """
        Add a message to the chat model messages, either as a user or as an assistant.
        :param message:
        :param role: 'user' or 'assistant'
        :return: None
        """
        self.messages.append({
            "role": role,
            "content": message
        })

    def check_appointment_confirmation(self):
        """
        Uses the chat model to check whether the user's selected appointment time is valid.
        :return: whether the appointment selection is valid
        """
        user_message = "Were you able to get a valid appointment date and time from my response that was " \
                       "part of the options you offered, yes or no?"
        self.add_message_to_chat(user_message, "user")

        response = openai.ChatCompletion.create(
            model=_CHAT_MODEL,
            messages=self.messages)
        response = response.choices[0].message.content.strip().strip(".")
        self.add_message_to_chat(response, "assistant")

        self.logger.debug("appointment confirmation gpt response: %s", response)

        if "yes" in response.lower():
            user_message = "Provide just the date and time that I selected, no extra tokens."
            self.add_message_to_chat(user_message, "user")

            response = openai.ChatCompletion.create(
                model=_CHAT_MODEL,
                messages=self.messages)
            response = response.choices[0].message.content.strip().strip(".")
            self.add_message_to_chat(response, "assistant")

            self.appointment_info.appointment_confirmed = True
            self.appointment_info.date_and_time = response

            return True

        else:
            return False

    def log_state(self, extra: Optional[str] = None):
        """
        Logging tool
        """
        if extra:
            self.logger.debug("extra: %s", extra)

        if self.conversation_stage == ConversationStage.TIME_SELECTION:
            self.logger.debug("In time selection, appointment info: %s", self.appointment_info)

        else:
            info_group_key = self.info_group_names[self.info_group_index]
            info_group = self.agent_config.info_groups[info_group_key]

            self.logger.debug("info group index: %s", self.info_group_index)
            self.logger.debug("info group preamble: %s, current element: %d",
                              info_group.preamble,
                              info_group.current_element)
            for element in info_group.info_elements:
                self.logger.debug("element name: %s, type: %s, value: %s", element.element_name,
                                  element.element_type, element.element_value)


class MedicalOfficeAgentFactory(AgentFactory):
    """
    Used by the TelephonyServer
    """
    def create_agent(self, agent_config: AgentConfig, logger: Optional[logging.Logger] = None) -> BaseAgent:
        return MedicalOfficeAgent(agent_config=agent_config, logger=logger)
