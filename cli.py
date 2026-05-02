from chatbot import Chatbot

class Conversation:
    def __init__(self, chatbot):
        self.chatbot = chatbot
        print(self.chatbot.onboard_prompt)

    def _update_status(self, status):
        if status == "running_tools":
            print(f"{self.chatbot.name}: thinking... (running tools)")
        elif status == "generating_final":
            print(f"{self.chatbot.name}: thinking... (writing answer)")

    def _update_output(self, output):
        print(f"{self.chatbot.name}: {output}")

    def _print_supporting_evidence(self, payload):
        citations = payload.get("citations", [])
        evidence = payload.get("evidence", [])
        retrieval = payload.get("retrieval") or {}

        if citations:
            print("Citations:")
            for citation in citations:
                print(f"- {citation.get('title', 'Untitled')} | {citation.get('url', '')}")

        if evidence:
            print("Evidence:")
            for item in evidence:
                matched_terms = ", ".join(item.get("matched_terms", [])) or "n/a"
                print(
                    f"- Rank {item.get('rank', '?')} | {item.get('title', 'Untitled')} | matched: {matched_terms}"
                )
                print(f"  {item.get('snippet', '')}")

        if retrieval:
            print(f"Retrieval confidence: {'low' if retrieval.get('low_confidence') else 'normal'}")

    def run(self):
        while True:
            user_input = input(self.chatbot.input_prompt)
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
            payload = self.chatbot.create_response(user_input, status_callback=self._update_status)
            self._update_output(payload["reply"])
            self._print_supporting_evidence(payload)

if __name__ == "__main__":
    chatbot = Chatbot()
    Conversation(chatbot).run()
