from chatbot.API import Chatbot

class Conversation: 
    def __init__(self, chatbot):
        self.chatbot = chatbot
        print(self.chatbot.onboard_prompt) # print the onboarding prompt when the conversation starts

    def run(self):
        while True:
            user_input = input(self.chatbot.input_prompt)
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
            response = self.chatbot.create_response(user_input)
            print(self.chatbot.history.messages[-1].content) # print the assistant's latest message content

if __name__ == "__main__":
    chatbot = Chatbot()
    Conversation(chatbot).run()