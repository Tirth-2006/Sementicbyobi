"""Few-shot prompt construction utilities."""
from typing import List, Dict


class PromptBuilder:
    """Build few-shot prompts for question answering."""

    def __init__(self, brief_instruction: str = None):
        """Initialize prompt builder.

        Args:
            brief_instruction: Instruction to add at the start of prompt
        """
        self.brief_instruction = brief_instruction or \
            "Answer the following question as briefly as possible."

    def format_single_example(self,
                             question: str,
                             answer: str = None,
                             context: str = None) -> str:
        """Format a single QA example.

        Args:
            question: Question text
            answer: Answer text (None for query example)
            context: Optional context text

        Returns:
            Formatted example string
        """
        prompt = ""

        if context:
            prompt += f"Context: {context}\n"

        prompt += f"Question: {question}\n"

        if answer is not None:
            prompt += f"Answer: {answer}\n\n"
        else:
            prompt += "Answer:"

        return prompt

    def build_few_shot_prompt(self, few_shot_examples: List[Dict]) -> str:
        """Build few-shot prompt from examples.

        Args:
            few_shot_examples: List of example dicts with 'question', 'answers', 'context'

        Returns:
            Complete few-shot prompt string
        """
        # Start with brief instruction
        prompt = self.brief_instruction + "\n\n"

        # Add each few-shot example
        for example in few_shot_examples:
            question = example['question']
            # Take first answer from the list
            answer = example['answers']['text'][0]
            context = example.get('context')

            prompt += self.format_single_example(question, answer, context)

        return prompt

    def build_query_prompt(self,
                          few_shot_prompt: str,
                          question: str,
                          context: str = None) -> str:
        """Build complete prompt for a query question.

        Args:
            few_shot_prompt: Pre-built few-shot prompt
            question: Query question
            context: Optional context for the question

        Returns:
            Complete prompt ready for model
        """
        query_part = self.format_single_example(question, answer=None, context=context)
        return few_shot_prompt + query_part


def create_prompt_builder(brief_type: str = "default") -> PromptBuilder:
    """Create a prompt builder with specified brief instruction type.

    Args:
        brief_type: Type of brief instruction ('default' or 'chat')

    Returns:
        PromptBuilder instance
    """
    brief_instructions = {
        'default': "Answer the following question as briefly as possible.",
        'chat': "Answer the following question in a single brief but complete sentence."
    }

    instruction = brief_instructions.get(brief_type, brief_instructions['default'])
    return PromptBuilder(brief_instruction=instruction)
