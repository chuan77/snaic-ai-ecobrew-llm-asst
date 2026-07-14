import logging

import gradio as gr

from guardrail.validate import validate_answer
from scripts.serve import get_predict_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ecobrew.guardrail")

predict = get_predict_fn()


def respond(message, history):
    raw_answer = predict(message)
    final_answer, overridden = validate_answer(message, raw_answer)
    if overridden:
        logger.info("guardrail overrode answer | question=%r raw_answer=%r", message, raw_answer)
    return final_answer


GREETING = (
    "Hi, I'm the EcoBrew Smart Coffee Maker Assistant. "
    "What would you like to know about EcoBrew — pricing, specs, warranty, or support?"
)

demo = gr.ChatInterface(
    fn=respond,
    chatbot=gr.Chatbot(value=[{"role": "assistant", "content": GREETING}]),
    title="EcoBrew Smart Coffee Maker Assistant",
    description="Ask about EcoBrew pricing, specs, warranty, and support. Closed-book — answers come only from injected training, not lookup.",
)

if __name__ == "__main__":
    demo.launch()
