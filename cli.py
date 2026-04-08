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

    def run(self):
        while True:
            user_input = input(self.chatbot.input_prompt)
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
            response = self.chatbot.create_response(user_input, status_callback=self._update_status)
            self._update_output(response)

if __name__ == "__main__":
    chatbot = Chatbot()
    Conversation(chatbot).run()