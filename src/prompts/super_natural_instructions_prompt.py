import random


class SuperNaturalInstructionPrompt:
    def __init__(self, example=0, add_definition=True, add_explanation=False, add_paraphrased_definition=False,
                 add_adversarial_definition=False):
        self._example = example
        self.add_definition = add_definition
        self.add_explanation = add_explanation
        self.add_paraphrased_definition = add_paraphrased_definition
        self.add_adversarial_definition = add_adversarial_definition

    def generate_context(self, instance):
        sample_size = min(self._example, len(instance['positive_examples']))
        sampled_examples = random.sample(range(len(instance['positive_examples'])), k=sample_size)
        context = instance['definition'] if self.add_definition else \
            (instance['paraphrased_definition'] if self.add_paraphrased_definition else
             (instance['adversarial_definition'] if self.add_adversarial_definition else ""))
        positive_examples = [
            "Input: " + instance['positive_examples'][idx]['input'] + "\n" +
            "Output: " + instance['positive_examples'][idx]['output'] + "\n" +
            ("Explanation: " + instance['positive_examples'][idx]['explanation'] if self.add_explanation else "")
            for idx in sampled_examples
        ]
        if len(positive_examples) > 0:
            context += "\n\nRefer to the following examples for reference:\n\n" + "\n\n".join(
                positive_examples)
        return context

    def get_prompt(self, batch):
        prompts = []
        label = []
        for idx in range(len(batch)):
            context = self.generate_context(batch[idx])
            current_instance = "Input: " + batch[idx]['input'] + "\n" + "Output: "
            full_prompt = context + "\n\n" + "Now complete the following task: \n" + current_instance
            prompts.append(full_prompt)
            label.append(batch[idx]['output'][0])
        return prompts, label
