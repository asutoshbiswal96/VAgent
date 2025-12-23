import os
from typing import Dict
from src.rag import SimpleRAG
from src.privacy import redact_record, insert_pii
from src.gemini_client import GeminiClient


class LICAgent:
    def __init__(self, csv_path: str):
        self.rag = SimpleRAG()
        self.rag.index_from_csv(csv_path)
        self.gemini = GeminiClient()

    def _build_prompt(self, redacted_record: Dict[str, str], history: str) -> str:
        # Provide LLM with redacted policy info and conversation history. PII placeholders present.
        lines = [
            "You are a polite LIC customer-service agent whose task is to remind customers about premium payments.",
            "Do NOT attempt to infer or guess personal information. Placeholders like [NAME], [PHONE], [EMAIL] will be kept as placeholders.",
            "Use friendly tone and confirm payment details when asked.",
            "---",
            "Policy holder information:\n",
        ]
        for k, v in redacted_record.items():
            lines.append(f"{k}: {v}")
        lines.append("---")
        if history:
            lines.append("Conversation history:")
            lines.append(history)
        lines.append("Agent:")
        return "\n".join(lines)

    def start_conversation(self, policy_id: str):
        docs = self.rag.retrieve(policy_id)
        if not docs:
            print("No policy found")
            return
        row = docs[0]
        redacted, mapping = redact_record(row)
        history = ""
        print(f"Starting conversation with policy {policy_id}. (PII will not be sent to LLM)")
        while True:
            user_input = input('Policyholder> ')
            if user_input.strip().lower() in ('exit', 'quit'):
                print('Ending conversation')
                break
            # First, check for simple local intents that should NOT be sent to the LLM
            handled, final_response, redacted_agent_reply = self._handle_local_request(user_input, row, mapping)
            # redact user input before adding to history
            redacted_user = user_input
            for ph, real in mapping.items():
                redacted_user = redacted_user.replace(real, ph)
            history += f"Policyholder: {redacted_user}\n"
            if handled:
                # append redacted agent reply to history (so LLM sees placeholders only)
                history += f"Agent: {redacted_agent_reply}\n"
                print('\nAgent> ' + final_response + '\n')
                continue

            # otherwise send redacted context + history to LLM
            prompt = self._build_prompt(redacted, history)
            agent_response = self.gemini.generate(prompt)
            # locally insert PII when "sending" to the user
            final_response = insert_pii(agent_response, mapping)
            # store redacted form of the agent response in history
            redacted_agent = agent_response
            for ph, real in mapping.items():
                redacted_agent = redacted_agent.replace(real, ph)
            history += f"Agent: {redacted_agent}\n"
            print('\nAgent> ' + final_response + '\n')

    def _handle_local_request(self, user_input: str, row: dict, mapping: dict):
        """Handle simple factual/PII requests locally.

        Returns (handled: bool, final_response_for_user: str, redacted_agent_reply: str)
        """
        ui = user_input.lower()
        # PII requests
        if 'email' in ui or 'e-mail' in ui:
            real = mapping.get('[EMAIL]') or row.get('email', '')
            if real:
                return True, f"Your registered email is: {real}", "Your registered email is: [EMAIL]"
        if 'phone' in ui or 'mobile' in ui or 'contact number' in ui:
            real = mapping.get('[PHONE]') or row.get('phone', '')
            if real:
                return True, f"Your registered phone number is: {real}", "Your registered phone number is: [PHONE]"
        if 'name' in ui or 'my name' in ui:
            real = mapping.get('[NAME]') or row.get('name', '')
            if real:
                return True, f"Your name on record is: {real}", "Your name on record is: [NAME]"
        # Policy/fact requests
        if 'due' in ui or 'due date' in ui:
            due = row.get('due_date', '')
            if due:
                return True, f"Your premium due date is: {due}", f"Your premium due date is: {due}"
        if 'premium' in ui or 'amount' in ui:
            amt = row.get('premium_amount', '')
            if amt:
                return True, f"Your premium amount is: {amt}", f"Your premium amount is: {amt}"
        return False, '', ''


if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'policyholders.csv')
    csv_path = os.path.abspath(csv_path)
    agent = LICAgent(csv_path)
    print('Loaded agent. Pick a policy id from data/policyholders.csv (e.g. P001)')
    pid = input('Policy id> ').strip()
    agent.start_conversation(pid)
