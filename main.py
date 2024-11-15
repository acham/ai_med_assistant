import logging
import os
import uvicorn
from dotenv import load_dotenv
from typing import Optional

from fastapi import FastAPI, Form
from fastapi.templating import Jinja2Templates
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.conversation.outbound_call import OutboundCall
from vocode.streaming.telephony.server.base import InboundCallConfig, TelephonyServer
from vocode.streaming.models.synthesizer import StreamElementsSynthesizerConfig

from agent.medical_office_agent import MedicalOfficeAgentFactory, get_initial_agent_config
from memory_config import config_manager


app = FastAPI(docs_url=None)
templates = Jinja2Templates(directory="templates")

_LOG_FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=_LOG_FORMAT)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

load_dotenv("env.env")

BASE_URL = os.getenv("BASE_URL")

TWILIO_CONFIG = TwilioConfig(
    account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
    auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
)

TWILIO_PHONE = os.getenv("OUTBOUND_CALLER_NUMBER")

CONFIG_MANAGER = config_manager  # RedisConfigManager()

SYNTH_CONFIG = StreamElementsSynthesizerConfig.from_telephone_output_device()

telephony_server = TelephonyServer(
    base_url=BASE_URL,
    config_manager=CONFIG_MANAGER,
    inbound_call_configs=[
        InboundCallConfig(url="/inbound_call",
                          agent_config=get_initial_agent_config(),
                          twilio_config=TWILIO_CONFIG,
                          synthesizer_config=SYNTH_CONFIG)
    ],
    agent_factory=MedicalOfficeAgentFactory(),
    logger=_logger
)
app.include_router(telephony_server.get_router())


def start_outbound_call(to_phone: Optional[str]):
    _logger.info(f"to_phone: {to_phone}")

    if to_phone:
        outbound_call = OutboundCall(base_url=BASE_URL,
                                     to_phone=to_phone,
                                     from_phone=TWILIO_PHONE,
                                     config_manager=CONFIG_MANAGER,
                                     agent_config=get_initial_agent_config(),
                                     twilio_config=TWILIO_CONFIG,
                                     synthesizer_config=SYNTH_CONFIG,
                                     logger=_logger)

        _logger.info(f"OutboundCall created")
        outbound_call.start()


@app.post("/start_outbound_call")
async def api_start_outbound_call(to_phone: Optional[str] = Form(None)):
    _logger.info("start outbound call endpoint")
    start_outbound_call(to_phone)
    return {"status": "success"}


@app.post("/handle_failure")
async def api_handle_failed_call():
    _logger.info("call failed, handling..")
    return {"status": "failure"}


def main():
    uvicorn.run(app, host="0.0.0.0", port=3000)


if __name__ == "__main__":
    main()
