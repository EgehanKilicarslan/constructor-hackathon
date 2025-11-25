import argparse
import random
from datasets import Dataset

# Mocking the Task configuration and methods for demonstration
class MockTaskConfig:
    def __init__(self):
        self.target_delimiter = " -> "
        self.fewshot_delimiter = "\n---\n"
        self.fewshot_split = "train"
        self.test_split = "test"
        self.doc_to_choice = None # Simplified for demo

class MockTask:
    def __init__(self):
        self._config = MockTaskConfig()

    def doc_to_text(self, doc):
        return doc["question"]

    def doc_to_target(self, doc):
        return doc["answer"]

    def doc_to_choice(self, doc):
        # Not used in this simplified demo, but kept for API compatibility
        return None

# Original ContextSampler class from samplers.py
class ContextSampler:
    def __init__(self, docs, task, fewshot_indices=None, rnd=None) -> None:
        self.rnd = rnd
        assert self.rnd, "must pass rnd to FewShotSampler!"

        self.task = task
        self.config = task._config

        self.target_delimiter = self.config.target_delimiter
        self.fewshot_delimiter = self.config.fewshot_delimiter

        self.doc_to_text = self.task.doc_to_text
        self.doc_to_target = self.task.doc_to_target
        self.doc_to_choice = self.task.doc_to_choice

        self.docs = docs  # HF dataset split, provided by task._fewshot_docs()
        if fewshot_indices:  # subset few-shot docs from
            self.docs = self.docs.select(fewshot_indices)

    def get_context(self, doc, num_fewshot):
        # draw an extra fewshot sample if using same split as evaluating on
        n_samples = num_fewshot + 1 if self.config.fewshot_split == self.config.test_split else num_fewshot

        # draw `n_samples` docs from fewshot_docs
        fewshotex = self.sample(n_samples)

        # get rid of the doc that's the one we're evaluating, if it's in the fewshot
        # TODO: should we just stop people from using fewshot from same split as evaluating?
        selected_docs = [x for x in fewshotex if x != doc][:num_fewshot]

        labeled_examples = (
            self.fewshot_delimiter.join(
                [
                    # TODO: is separating doc_to_text and doc_to_target by one space always desired?
                    (self.doc_to_text(doc) if (self.config.doc_to_choice is None or type(self.doc_to_text(doc)) is str) else self.doc_to_choice(doc)[self.doc_to_text(doc)])
                    + self.target_delimiter
                    + (
                        str(self.doc_to_target(doc)[0])
                        if type(self.doc_to_target(doc)) is list
                        else self.doc_to_target(doc) if (self.config.doc_to_choice is None or type(self.doc_to_target(doc)) is str) else str(self.doc_to_choice(doc)[self.doc_to_target(doc)])
                    )
                    for doc in selected_docs
                ]
            )
            + self.fewshot_delimiter
        )

        return labeled_examples

    def sample(self, n):
        """
        Draw `n` samples from our fewshot docs. This method should be overridden by subclasses.
        """
        return self.rnd.sample(self.docs, n)

# Original FirstNSampler class from samplers.py
class FirstNSampler(ContextSampler):
    def sample(self, n) -> None:
        """
        Draw the first `n` samples in order from the specified split.
        Used for tasks with "canonical" ordered fewshot examples, such as MMLU and CMMLU.
        """
        assert n <= len(self.docs), f"Error: number of fewshot samples requested exceeds the {len(self.docs)} that are available."
        return self.docs[:n]

def run_sampler_demo(num_fewshot: int = 2, seed: int = 42):
    """
    Demonstrates the ContextSampler and FirstNSampler functionality.
    """
    print(f"Running sampler demo with {num_fewshot} few-shot examples and seed {seed}\n")

    # 1. Prepare mock data
    mock_data = {
        "id": [f"doc_{i}" for i in range(10)],
        "question": [f"What is the capital of country {i}?" for i in range(10)],
        "answer": [f"Capital {i}" for i in range(10)],
    }
    mock_dataset = Dataset.from_dict(mock_data)

    # 2. Instantiate mock task
    mock_task = MockTask()

    # 3. Demonstrate ContextSampler (random sampling)
    print("--- ContextSampler (Random Sampling) ---")
    rnd_context_sampler = random.Random(seed)
    sampler_random = ContextSampler(mock_dataset, mock_task, rnd=rnd_context_sampler)

    # Pick a document to evaluate (e.g., doc_5)
    doc_to_evaluate_random = mock_dataset[5]
    print(f"Document to evaluate: {doc_to_evaluate_random['question']} -> {doc_to_evaluate_random['answer']}")

    context_random = sampler_random.get_context(doc_to_evaluate_random, num_fewshot)
    print("\nGenerated Context:")
    print(context_random)

    # 4. Demonstrate FirstNSampler (ordered sampling)
    print("\n--- FirstNSampler (Ordered Sampling) ---")
    # FirstNSampler doesn't strictly need a random object for its `sample` method,
    # but the base class constructor requires it. We can pass a dummy one.
    rnd_first_sampler = random.Random(seed)
    sampler_first_n = FirstNSampler(mock_dataset, mock_task, rnd=rnd_first_sampler)

    # Pick a document to evaluate (e.g., doc_5)
    doc_to_evaluate_first_n = mock_dataset[5]
    print(f"Document to evaluate: {doc_to_evaluate_first_n['question']} -> {doc_to_evaluate_first_n['answer']}")

    context_first_n = sampler_first_n.get_context(doc_to_evaluate_first_n, num_fewshot)
    print("\nGenerated Context:")
    print(context_first_n)

    print("\nDemo complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstrate LLaVA-MORE sampling logic.")
    parser.add_argument("--num-fewshot", type=int, default=2,
                        help="Number of few-shot examples to include in the context.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling.")
    args = parser.parse_args()
    run_sampler_demo(args.num_fewshot, args.seed)
