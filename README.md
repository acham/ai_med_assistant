# A Guided AI Voice Agent
This shows an approach to leveraging LLMs to build AI agents with specific goals.

The code implements a voice agent for a medical office assistant and uses the OpenAI 
Chat Completions API to help collect and validate specific information from patients.

`main.py` is based on [Vocode's phone call server template](https://docs.vocode.dev/open-source-quickstart).

All requirements with correct versions are in requirements.txt; requires python 3.10.* to run, tested with 3.10.14.

To run, this implementation requires a dedicated phone number with Twilio configured with a webhook to the `inbound_call` URL where the service will be running.
The corresponding environment variables will have to be specified in `env.env`.
