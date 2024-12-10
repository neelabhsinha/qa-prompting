class QuestionAnsweringPrompt:
    def __init__(self):
        self.questions = {
            "topic": "What is the main topic or focus of the content?",
            "key_pts": "What are the key points or arguments presented?",
            "entities": "Who are the three main entities or individuals involved, and what roles do they play?",
            "timeline": "Which timeline, if any, is being discussed here?",
            "details": "What are the supporting details, examples, or evidence provided?",
            "conclude": "What conclusions, recommendations, impacts, or implications are mentioned, if any?",
            "tone": "What is the overall tone or sentiment (e.g., objective, critical, positive, negative, etc.)?",
            "challenges": "What questions or challenges does the content raise?",
            "insights": "What unique insights or perspectives are offered?",
            "audience": "What audience is the content aimed at, and how does this affect its presentation?"
        }

    def get_prompt(self, texts):
        prompts = []
        for text in texts:
            for key, value in self.questions.items():
                prompt = ("Given the following text:\n" + text +
                          "\n\nAnswer the following question precisely without any additional detail: ")
                prompt += f"{value}\n"
                prompt += "Answer:"
                prompts.append(prompt)
        return prompts
