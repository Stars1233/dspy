---
sidebar_position: 1
---

# DSPy Optimizers (formerly Teleprompters)


A **DSPy optimizer** is an algorithm that can tune the parameters of a DSPy program (i.e., the prompts and/or the LM weights) to maximize the metrics you specify, like accuracy.


A typical DSPy optimizer takes three things:

- Your **DSPy program**. This may be a single module (e.g., `dspy.Predict`) or a complex multi-module program.

- Your **metric**. This is a function that evaluates the output of your program, and assigns it a score (higher is better).

- A few **training inputs**. This may be very small (i.e., only 5 or 10 examples) and incomplete (only inputs to your program, without any labels).

If you happen to have a lot of data, DSPy can leverage that. But you can start small and get strong results.

**Note:** Formerly called teleprompters. We are making an official name update, which will be reflected throughout the library and documentation.


## What does a DSPy Optimizer tune? How does it tune them?

Different optimizers in DSPy will tune your program's quality by **synthesizing good few-shot examples** for every module, like `dspy.BootstrapRS`,<sup>[1](https://arxiv.org/abs/2310.03714)</sup> **proposing and intelligently exploring better natural-language instructions** for every prompt, like `dspy.MIPROv2`,<sup>[2](https://arxiv.org/abs/2406.11695)</sup> and **building datasets for your modules and using them to finetune the LM weights** in your system, like `dspy.BootstrapFinetune`.<sup>[3](https://arxiv.org/abs/2407.10930)</sup>

??? "What's an example of a DSPy optimizer? How do different optimizers work?"

    Take the `dspy.MIPROv2` optimizer as an example. First, MIPRO starts with the **bootstrapping stage**. It takes your program, which may be unoptimized at this point, and runs it many times across different inputs to collect traces of input/output behavior for each one of your modules. It filters these traces to keep only those that appear in trajectories scored highly by your metric. Second, MIPRO enters its **grounded proposal stage**. It previews your DSPy program's code, your data, and traces from running your program, and uses them to draft many potential instructions for every prompt in your program. Third, MIPRO launches the **discrete search stage**. It samples mini-batches from your training set, proposes a combination of instructions and traces to use for constructing every prompt in the pipeline, and evaluates the candidate program on the mini-batch. Using the resulting score, MIPRO updates a surrogate model that helps the proposals get better over time.

    One thing that makes DSPy optimizers so powerful is that they can be composed. You can run `dspy.MIPROv2` and use the produced program as an input to `dspy.MIPROv2` again or, say, to `dspy.BootstrapFinetune` to get better results. This is partly the essence of `dspy.BetterTogether`. Alternatively, you can run the optimizer and then extract the top-5 candidate programs and build a `dspy.Ensemble` of them. This allows you to scale _inference-time compute_ (e.g., ensembles) as well as DSPy's unique _pre-inference time compute_ (i.e., optimization budget) in highly systematic ways.



## What DSPy Optimizers are currently available?

Optimizers can be accessed via `from dspy.teleprompt import *`.

### Automatic Few-Shot Learning

These optimizers extend the signature by automatically generating and including **optimized** examples within the prompt sent to the model, implementing few-shot learning.

1. [**`LabeledFewShot`**](../../api/optimizers/LabeledFewShot.md): Simply constructs few-shot examples (demos) from provided labeled input and output data points.  Requires `k` (number of examples for the prompt) and `trainset` to randomly select `k` examples from.

2. [**`BootstrapFewShot`**](../../api/optimizers/BootstrapFewShot.md): Uses a `teacher` module (which defaults to your program) to generate complete demonstrations for every stage of your program, along with labeled examples in `trainset`. Parameters include `max_labeled_demos` (the number of demonstrations randomly selected from the `trainset`) and `max_bootstrapped_demos` (the number of additional examples generated by the `teacher`). The bootstrapping process employs the metric to validate demonstrations, including only those that pass the metric in the "compiled" prompt. Advanced: Supports using a `teacher` program that is a *different* DSPy program that has compatible structure, for harder tasks.

3. [**`BootstrapFewShotWithRandomSearch`**](../../api/optimizers/BootstrapFewShotWithRandomSearch.md): Applies `BootstrapFewShot` several times with random search over generated demonstrations, and selects the best program over the optimization. Parameters mirror those of `BootstrapFewShot`, with the addition of `num_candidate_programs`, which specifies the number of random programs evaluated over the optimization, including candidates of the uncompiled program, `LabeledFewShot` optimized program, `BootstrapFewShot` compiled program with unshuffled examples and `num_candidate_programs` of `BootstrapFewShot` compiled programs with randomized example sets.

4. [**`KNNFewShot`**](../../api/optimizers/KNNFewShot.md). Uses k-Nearest Neighbors algorithm to find the nearest training example demonstrations for a given input example. These nearest neighbor demonstrations are then used as the trainset for the BootstrapFewShot optimization process.


### Automatic Instruction Optimization

These optimizers produce optimal instructions for the prompt and, in the case of MIPROv2 can also optimize the set of few-shot demonstrations.

5. [**`COPRO`**](../../api/optimizers/COPRO.md): Generates and refines new instructions for each step, and optimizes them with coordinate ascent (hill-climbing using the metric function and the `trainset`). Parameters include `depth` which is the number of iterations of prompt improvement the optimizer runs over.

6. [**`MIPROv2`**](../../api/optimizers/MIPROv2.md): Generates instructions *and* few-shot examples in each step. The instruction generation is data-aware and demonstration-aware. Uses Bayesian Optimization to effectively search over the space of generation instructions/demonstrations across your modules.


### Automatic Finetuning

This optimizer is used to fine-tune the underlying LLM(s).

7. [**`BootstrapFinetune`**](../../api/optimizers/BootstrapFinetune.md): Distills a prompt-based DSPy program into weight updates. The output is a DSPy program that has the same steps, but where each step is conducted by a finetuned model instead of a prompted LM.


### Program Transformations

8. [**`Ensemble`**](../../api/optimizers/Ensemble.md): Ensembles a set of DSPy programs and either uses the full set or randomly samples a subset into a single program.


## Which optimizer should I use?

Ultimately, finding the ‘right’ optimizer to use & the best configuration for your task will require experimentation. Success in DSPy is still an iterative process - getting the best performance on your task will require you to explore and iterate.  

That being said, here's the general guidance on getting started:

- If you have **very few examples** (around 10), start with `BootstrapFewShot`.
- If you have **more data** (50 examples or more), try  `BootstrapFewShotWithRandomSearch`.
- If you prefer to do **instruction optimization only** (i.e. you want to keep your prompt 0-shot), use `MIPROv2` [configured for 0-shot optimization](../../api/optimizers/MIPROv2.md). 
- If you’re willing to use more inference calls to perform **longer optimization runs** (e.g. 40 trials or more), and have enough data (e.g. 200 examples or more to prevent overfitting) then try `MIPROv2`. 
- If you have been able to use one of these with a large LM (e.g., 7B parameters or above) and need a very **efficient program**, finetune a small LM for your task with `BootstrapFinetune`.

## How do I use an optimizer?

They all share this general interface, with some differences in the keyword arguments (hyperparameters). A full list can be found in the [API reference](../../api/optimizers/BetterTogether.md).

Let's see this with the most common one, `BootstrapFewShotWithRandomSearch`.

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 8-shot examples of your program's steps.
# The optimizer will repeat this 10 times (plus some initial attempts) before selecting its best attempt on the devset.
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4, num_candidate_programs=10, num_threads=4)

teleprompter = BootstrapFewShotWithRandomSearch(metric=YOUR_METRIC_HERE, **config)
optimized_program = teleprompter.compile(YOUR_PROGRAM_HERE, trainset=YOUR_TRAINSET_HERE)
```


!!! info "Getting Started III: Optimizing the LM prompts or weights in DSPy programs"
    A typical simple optimization run costs on the order of $2 USD and takes around ten minutes, but be careful when running optimizers with very large LMs or very large datasets.
    Optimizer runs can cost as little as a few cents or up to tens of dollars, depending on your LM, dataset, and configuration.
    
    === "Optimizing prompts for a ReAct agent"
        This is a minimal but fully runnable example of setting up a `dspy.ReAct` agent that answers questions via
        search from Wikipedia and then optimizing it using `dspy.MIPROv2` in the cheap `light` mode on 500
        question-answer pairs sampled from the `HotPotQA` dataset.

        ```python linenums="1"
        import dspy
        from dspy.datasets import HotPotQA

        dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

        def search(query: str) -> list[str]:
            """Retrieves abstracts from Wikipedia."""
            results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
            return [x['text'] for x in results]

        trainset = [x.with_inputs('question') for x in HotPotQA(train_seed=2024, train_size=500).train]
        react = dspy.ReAct("question -> answer", tools=[search])

        tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24)
        optimized_react = tp.compile(react, trainset=trainset)
        ```

        An informal run similar to this on DSPy 2.5.29 raises ReAct's score from 24% to 51%.

    === "Optimizing prompts for RAG"
        Given a retrieval index to `search`, your favorite `dspy.LM`, and a small `trainset` of questions and ground-truth responses, the following code snippet can optimize your RAG system with long outputs against the built-in `dspy.SemanticF1` metric, which is implemented as a DSPy module.

        ```python linenums="1"
        class RAG(dspy.Module):
            def __init__(self, num_docs=5):
                self.num_docs = num_docs
                self.respond = dspy.ChainOfThought('context, question -> response')

            def forward(self, question):
                context = search(question, k=self.num_docs)   # not defined in this snippet, see link above
                return self.respond(context=context, question=question)

        tp = dspy.MIPROv2(metric=dspy.SemanticF1(), auto="medium", num_threads=24)
        optimized_rag = tp.compile(RAG(), trainset=trainset, max_bootstrapped_demos=2, max_labeled_demos=2)
        ```

        For a complete RAG example that you can run, start this [tutorial](../../tutorials/rag/index.ipynb). It improves the quality of a RAG system over a subset of StackExchange communities from 53% to 61%.

    === "Optimizing weights for Classification"
        <details><summary>Click to show dataset setup code.</summary>

        ```python linenums="1"
        import random
        from typing import Literal

        from datasets import load_dataset

        import dspy
        from dspy.datasets import DataLoader

        # Load the Banking77 dataset.
        CLASSES = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True).features["label"].names
        kwargs = {"fields": ("text", "label"), "input_keys": ("text",), "split": "train", "trust_remote_code": True}

        # Load the first 2000 examples from the dataset, and assign a hint to each *training* example.
        trainset = [
            dspy.Example(x, hint=CLASSES[x.label], label=CLASSES[x.label]).with_inputs("text", "hint")
            for x in DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)[:2000]
        ]
        random.Random(0).shuffle(trainset)
        ```
        </details>

        ```python linenums="1"
        import dspy
        dspy.configure(lm=dspy.LM('openai/gpt-4o-mini-2024-07-18'))

        # Define the DSPy module for classification. It will use the hint at training time, if available.
        signature = dspy.Signature("text, hint -> label").with_updated_fields('label', type_=Literal[tuple(CLASSES)])
        classify = dspy.ChainOfThought(signature)

        # Optimize via BootstrapFinetune.
        optimizer = dspy.BootstrapFinetune(metric=(lambda x, y, trace=None: x.label == y.label), num_threads=24)
        optimized = optimizer.compile(classify, trainset=trainset)

        optimized(text="What does a pending cash withdrawal mean?")
        ```

        **Possible Output (from the last line):**
        ```text
        Prediction(
            reasoning='A pending cash withdrawal indicates that a request to withdraw cash has been initiated but has not yet been completed or processed. This status means that the transaction is still in progress and the funds have not yet been deducted from the account or made available to the user.',
            label='pending_cash_withdrawal'
        )
        ```

        An informal run similar to this on DSPy 2.5.29 raises GPT-4o-mini's score 66% to 87%.


## Saving and loading optimizer output

After running a program through an optimizer, it's useful to also save it. At a later point, a program can be loaded from a file and used for inference. For this, the `load` and `save` methods can be used.

```python
optimized_program.save(YOUR_SAVE_PATH)
```

The resulting file is in plain-text JSON format. It contains all the parameters and steps in the source program. You can always read it and see what the optimizer generated. You can add `save_field_meta` to additionally save the list of fields with the keys, `name`, `field_type`, `description`, and `prefix` with: `optimized_program.save(YOUR_SAVE_PATH, save_field_meta=True).


To load a program from a file, you can instantiate an object from that class and then call the load method on it.

```python
loaded_program = YOUR_PROGRAM_CLASS()
loaded_program.load(path=YOUR_SAVE_PATH)
```

