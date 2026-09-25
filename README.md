# 42_M03: Call Me Maybe

## Table of Contents
- [Description](#description)
- [Resources](#resources)
- [Instructions](#instructions)
	- [Makefile](#makefile)
	- [Manual](#manual)
	- [Decoder Class Usage](#decoder-class-usage)
- [Implementation](#implementation)
	- [Algorithm](#algorithm)
	- [Design](#design)
	- [Performance](#performance)
	- [Challenges](#challenges)
	- [Testing](#testing)

___

## Description
The goal of this project was to create a program that provided with prompt and list of available functions uses LLM and constrained decoding to return 100% valid JSON file with function call.

## Resources
- The `llm_sdk` module was provided by 42 school. Eventually I may create my own harness, but for now I enclose this one to be able to run the program.
- The development and various concepts used were consulted with AI (ChatGPT, Claude, Copilot)

## Instructions

### Makefile

```bash
make install					# install dependencies via uv

make / make all / make run		# run program

make debug						# run in a debug mode
make debug-print				# run program and show debug prints

make lint						# check the code with flake8 and mypy
make lint-strict				# check the code with flake8 and mypy in a strict mode

make clean						# delete temporary files and caches
make re							# make clean + run

make help						# show all MAKE options
```

### Manual
```
uv run python -m src
```
or if you want to specify the input and/or output files using flags:

```
uv run python -m src \
--functions_definition data/input/functions_definition.json \
--input data/input/function_calling_tests.json \
--output data/output/function_calls.json
```

### Decoder Class Usage

```python
# 1 - Decoder uses its default LLM (Qwen/Qwen3-0.6B)
decoder = Decoder()

# 2 - Decoder uses LLM model passed as argument
	# Required methods:
    # - get_path_to_vocab_file(self) -> str,
    # - encode(self, text: str) -> torch.Tensor | list[int],
	# - decode(self, ids: torch.Tensor | list[int]) -> str,
    # - get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]
decoder = Decoder(model=model)

# generate JSON output with function call
output = decoder.decode(function_defs, test_prompt)

# only select function to use and return the Functiondefinition object
decoder.select_function(function_defs, test_prompt)
```

## Implementation

### Algorithm
This program uses `constrained decoding` as a main tool to constrain the LLM during the token generation process to only make certain options available according to the required output format.

To recognize which tokens are the valid ones, we need to know where we are in the process so this is where the `DecoderState` comes. I defined all the substates in the output generation and predefined what is valid for them.

For example the first state is `DecoderState.START`:

```python
S = DecoderState

S.START: {
        "value": '{',
        "next_state": S.AFTER_OPEN_BRACE,
        "forced": True,
        "message": "Starting decoder",
    }
```
The only available value the LLM can generate is `{` aka the 'open brace'.

But how the LLM knows that?

From `decode` method we go to the `_navigate_state` and through that to `_model_generation`. Here, we ask the model to `get_logits_from_input_ids`
to get all the possible tokens with their probability scoring.

And THEN we generate the only constrained valid options by using `_get_valid_tokens`. In this case we just use the predefined one.

Finally we use masking and choose only from not masked logits:

```python
    def _model_generation(self, functions: list[FunctionDefinition],
            			  original_prompt: str) -> int:
        """
        Ask model for next token, mask invalid ones,
        pick best, advance state.
        """

        # return probabilities of tokens in vocabulary
        logits = np.array(
            self.model.get_logits_from_input_ids(self._input_ids)
        )

        valid_ids = self._get_valid_tokens(functions, original_prompt)

		# ...

        # create mask for each token
        mask = np.full(len(logits), True)

        # turn the mask off for valid tokens
        mask[valid_ids] = False

        # set invalid tokens to -infinity, so they are not chosen
        logits[mask] = -np.inf

        # pick the token with highest probability (valid)
        next_token = int(np.argmax(logits))

        return next_token
```

But some states are not that easy as they have more possible valid tokens.

They either branch (if there is other parameter following, generate comma, if not, generate closing brace) or they are predefined only in their type, but not the content (selected function name and its parameter values). These are delimited by the `_get_valid_tokens` method by the `DecoderState`.

Once we have the token generated we append it to the `_input_ids` that we send to the LLM in a loop so in each next state the input is updated with what we already generated in the previous one.

Exceptions are the states that generate the parameter values (`S.GENERATING_BOOLEAN`, `S.GENERATING_NUMBER`, `S.GENERATING_STRING`) where we check if the parameter is complete and if neccessary, trim the excess characters that may be in the token together with the ending character.

And finally we move to the next state.

Once finished, we validate the output and then use `json.loads` to export it.

All possible errors along the way are resolved by raising with custom message and catched in the main so the program does not crash.

### Design

#### Input
For parsing the CLI arguments I use `argparse` package and my own function `def parse_args() -> argparse.Namespace`.

Then I load the input files into variables using `load_function_definitions(path: str) -> list[FunctionDefinition]` and `load_test_prompts(path: str) -> list[TestPrompt]`.

For validating the right formats during the loading i use `pydantic` `BaseModel` combined with `Enum`.

#### Decoding
For decoding I use class `Decoder` and its method `decode` which returns validated JSON load.

I let the `select_function` method be also available to the user, other methods and attributes of the class are private as they only help during the process of decoding.

More about the constrained decoding approach in section [Algorithm](#algorithm).

#### Output

For writing the result into the output file I use `json` package and my own `write_outputs(output_objects: list[Any], path: str, overwrite: bool = False) -> None`.

It writes the output in default or user-defined output file. If it does not exist, it creates it.

I implemented a guard for overwriting already existing file. For the evaluation purpose I let it only write warning in the terminal so it does not stop automated testing. But I have interactive version ready in the comment.

> [!IMPORTANT]
> The results are written to the output file in the end of the process after all of them are generated and validated without issue. If there is any error, **the output file is not created at all!**
> The individual results of each test prompt are displayed on the stdout during the generation process.


### Performance
- JSON format: 22/22 tested cases (100%)
- LLM response:
	- Chosen function:valid in 20/22 of tested cases (91%)
	- Parameters: valid in 19/22 of tested cases (86%)
- Speed: approx. 15-50 seconds per prompt (depending on device and prompt)

When handed valid input, the response is pretty good. When the prompt does not contain all the neccessary information, the LLM hallucinates the output, but that is something that is not possible to solve without restraining the valid responses too. More about that in sections [Challenges](#challenges) and [Testing](#testing).

For better speed I precomputed the prefined tokens for JSON format `DecoderState` states so its not encoded over and over again.

To enhance the performance more I added KV caching by modifying model's `get_logits_from_input_ids` method so the input ids are not fed to the model again and again but they stay cached and each calling only adds the last previously generated token.

### Challenges

1) I had a hard time to take a grasp on the LLM concepts like tokenization, logits etc. Also to separate the generation from constrained decoding part.
- The more and more I had to fix all the next issues the more I actually understood what the program and its methods actually do.

2) The program was choosing the functions badly.
- Solved by prechoosing the function before generating the other stuff AND scoring it cumulatively after more tokens, not just the first one. I ended injecting the selected function during the generation itself so I do not do it before the start anymore. But I still use separate scoring and slightly different prompt for the function selection.

3) The program was generating the parameters badly.
There were technical issues in the states conditioning and ending the parameter generation.

4) I found out that my solution is not really according to the subject because it does not return the generated output but only the function name and parameters and the JSON is generated by `json.dump` in the writer function.
Had to redo quite some of the logic and found out that the export is not correct in some of the places (quote sign duplicities etc.).

I actually have broken the code a lot of times after I thought it is ok and changed just a little thing.

The biggest takeaway from this for me is to use branches in my repository to ensure I have a quick way out from the broken version. :D

### Testing
I created the code the way so I can run it as soon as possible and test it manually on the testing prompts. Sometimes I let AI test it with its own regression tests when debugging.

But mostly I ran the test prompts manually and checked what was generated. Especially with wrong functions or parameters chosen it was up to my eyes. I also used my custom prompts and function definitions sets to test free speach and ambiguous prompts and prompts that lead to no semantically valid conclusion.

Thanks to this I added a possibility of empty parameters in a function definition.

I also ruminated a lot about the prompts with no valid solution. I tried really hard to make the LLM say it is not able to solve the case but it kept hallucinating. I came to conclusion that it is not possible to restrain it without damaging the responses to the valid cases too.

---

## 👤 Author

**Simona Sucha**
*(also known as ssucha or vincent_syma)* <br>
Python & C · Software Developer · 42 student

🖥️ GitHub: https://github.com/vincent-syma/ <br>
🔗 LinkedIn: https://www.linkedin.com/in/simona-such%C3%A1-5a1b1928b <br>
✉️ Email: vincent.f.syma@email.cz <br>