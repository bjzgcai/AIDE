prompt_for_instruction_task_discovery = """
I have a dataset, and I would like you to brainstorm how it can be used in scientific tasks for instruction tuning of LLM.
The task should be compatible with all the samples in the dataset and can be organized as input-output pairs.
Your output should be a JSON array where each item describes a potential task for instruction tuning. Each item in the JSON array should include:
- **'task'**: A short description of the proposed task.
- **'input columns'**: A list of columns that will be used as input for the task.
- **'output columns'**: A list of columns that will be used as output for the task.
The column names MUST exist in the given dataset.
ONLY return the result in the following JSON format:
[
    {"task": "task1", "input columns": ["column1", "column2", ...], "output columns": ["column1", "column2", ...]},
    {"task": "task2", "input columns": ["column1", "column2", ...], "output columns": ["column1", "column2", ...]}
]
If you think there are no valid tasks (e.g., the dataset only contains one column, etc.), please return an empty JSON array [].
Here are some examples:
## Example 1
### Input:
1. **Dataset Name**: clinicalnlplab/BC5CDR_chemical_train
2. **Description**:
3. **Columns**: unprocessed, processed
4. **Sample Data**: Here are a few rows from the dataset:
    {"unprocessed": "In the sentence extracted from biomedical literature, identify all the chemical entities. The required answer format is the same sentence with HTML <span> tags to mark up specific entities. \n### Entity Markup Guides:\nUse <span class=\"chemical\"> to denote a chemical.\n\n### Input Text: Selegiline - induced postural hypotension in Parkinson ' s disease : a longitudinal study on the effects of drug withdrawal .\n### Output Text:", "processed": "<span class=\"chemical\">Selegiline</span> - induced postural hypotension in Parkinson ' s disease : a longitudinal study on the effects of drug withdrawal . "}
{"unprocessed": "In the sentence extracted from biomedical literature, identify all the chemical entities. The required answer format is the same sentence with HTML <span> tags to mark up specific entities. \n### Entity Markup Guides:\nUse <span class=\"chemical\"> to denote a chemical.\n\n### Input Text: OBJECTIVES : The United Kingdom Parkinson ' s Disease Research Group ( UKPDRG ) trial found an increased mortality in patients with Parkinson ' s disease ( PD ) randomized to receive 10 mg selegiline per day and L - dopa compared with those taking L - dopa alone .\n### Output Text:", "processed": "OBJECTIVES : The United Kingdom Parkinson ' s Disease Research Group ( UKPDRG ) trial found an increased mortality in patients with Parkinson ' s disease ( PD ) randomized to receive 10 mg <span class=\"chemical\">selegiline</span> per day and <span class=\"chemical\">L - dopa</span> compared with those taking <span class=\"chemical\">L - dopa</span> alone . "}
{"unprocessed": "In the sentence extracted from biomedical literature, identify all the chemical entities. The required answer format is the same sentence with HTML <span> tags to mark up specific entities. \n### Entity Markup Guides:\nUse <span class=\"chemical\"> to denote a chemical.\n\n### Input Text: Recently , we found that therapy with selegiline and L - dopa was associated with selective systolic orthostatic hypotension which was abolished by withdrawal of selegiline .\n### Output Text:", "processed": "Recently , we found that therapy with <span class=\"chemical\">selegiline</span> and <span class=\"chemical\">L - dopa</span> was associated with selective systolic orthostatic hypotension which was abolished by withdrawal of <span class=\"chemical\">selegiline</span> . "}
{"unprocessed": "In the sentence extracted from biomedical literature, identify all the chemical entities. The required answer format is the same sentence with HTML <span> tags to mark up specific entities. \n### Entity Markup Guides:\nUse <span class=\"chemical\"> to denote a chemical.\n\n### Input Text: This unwanted effect on postural blood pressure was not the result of underlying autonomic failure .\n### Output Text:", "processed": "This unwanted effect on postural blood pressure was not the result of underlying autonomic failure . "}
{"unprocessed": "In the sentence extracted from biomedical literature, identify all the chemical entities. The required answer format is the same sentence with HTML <span> tags to mark up specific entities. \n### Entity Markup Guides:\nUse <span class=\"chemical\"> to denote a chemical.\n\n### Input Text: The aims of this study were to confirm our previous findings in a separate cohort of patients and to determine the time course of the cardiovascular consequences of stopping selegiline in the expectation that this might shed light on the mechanisms by which the drug causes orthostatic hypotension .\n### Output Text:", "processed": "The aims of this study were to confirm our previous findings in a separate cohort of patients and to determine the time course of the cardiovascular consequences of stopping <span class=\"chemical\">selegiline</span> in the expectation that this might shed light on the mechanisms by which the drug causes orthostatic hypotension . "}
### Output:
[
    {"task": "named entity recognition", "input columns": ["unprocessed"], "output columns": ["processed"]}
]
## Example 2
### Input:
1. **Dataset Name**: AI4Chem/ChemBench4K
2. **Description**: chembench is a large-scale chemistry competency evaluation benchmark for language models, which includes nine chemistry core tasks and 4100 high-quality single-choice questions and answers.
3. **Columns**: B, D, A, C, answer, question
4. **Sample Data**: Here are a few rows from the dataset:
{"B": "In response, I offer C(C(CCCNC(=N)N)NC(C1NC(C2CCC(OCC3NC(=O)C4CCCN4C(C(NC(C(CCCCNC(=O)C1)NC(C(NC(C(C(C)OC(=O)CC(NC3=O)C(=O)NC(CC(C)C)C(N2)=O)NC(=O)C(NC(=O)C1N(CCC1)C(C(C(C)C)NC(=O)C(CCC(=O)O)NC(=O)C(N)C(C)O)=O)Cc1c[nH]c2c1cccc2)=O)Cc1ccc(cc1)O)=O)=O)Cc1ccccc1)=O)=O)=O)=O)(O)=O", "D": "The solution I propose is C(NC1C2N(C(C(=O)O)=C(CSc3nnnn3CCN(C)C)CS2)C1=O)(Cc1nc(N)sc1)=O.Cl.Cl", "A": "I believe the answer is OC(C(C)C(C(C)C=C(C)C=CCC(C)C=C(C=CC1CC=CC(O1)=O)C)=O)C(C)CC(C)=CC(=O)O", "answer": "C", "question": "Create a molecule that satisfies the conditions outlined in the description.\nThe molecule is a natural product found in Penicillium crustosum and Penicillium solitum with data available.", "C": "The solution I propose is C=C(C)C1OC2CCC3(C)C4(C)C(CCC3(O)C32C(O3)C1O)C1OC(C)(C2CC3C2(c2c(CC3=C)ccc3[nH]c4c1c23)O)C"}
{"B": "In my opinion, the answer would be c1c(c(=O)c2ccc(O)cc2o1)-c1cc(c(c(O)c1O)OC)C(C)(C=C)C", "D": "I would conclude that c1(cc(oc(c1C)=O)C=CC(C)=CC(C)=Cc1ccccc1)OC", "A": "In my opinion, the answer would be C12C3(O)CCC(C4=CC(=O)OC4)C3(C(O)CC1C1(C)CCC(OC3OC(C)C(C(C3)O)OC3CC(C(C(O3)C)OC3CC(C(C(O3)C)O)O)O)CC1CC2)C", "answer": "C", "question": "Create a molecule that satisfies the conditions outlined in the description.\nThe molecule is a natural product found in Cinnamomum camphora, Otoba parvifolia, and Zanthoxylum wutaiense with data available.", "C": "From my perspective, the answer is O1C(C2C(C1)C(c1cc3c(OCO3)cc1)OC2)c1ccc(c(c1)OC)O"}
{"B": "As for my answer, it is c12cc3OCOc3c(OC)c1C(OC(C2)c1ccccc1)=O", "D": "I would say that CC(OC1C2(C(CC(OC)C2C(CC2C1C(=C)C(O2)=O)C)=O)C)=O", "A": "The solution I propose is C(=C)(C1OC1c1c(ccc2c1oc(cc2)=O)OCC=C(C)C)C", "answer": "B", "question": "Synthesize a molecule that matches the given characteristics.\nThe molecule is a natural product found in Hammada scoparia with data available.", "C": "The solution I propose is C(O)(CC(CCCCCCCCC)O)=O"}
{"B": "My response is OC1C(CC(O)(C(OC(CC(=O)OCC=Cc2cc(O)c(OC3C(C(C(O)C(O3)CO)O)O)c(OC)c2)(CC(O)=O)C)=O)CC1O)OC(=O)C", "D": "The solution I propose is O=C1C2C3C=CC(C4CC43)C2C(=O)N1N=Cc1cccs1", "A": "I believe the answer is N12C(=O)C=CC32CCC2(C(C3)C(OC)=O)C(c3c(cccc3)N2C(OC)=O)(C(=O)OC)CC1", "answer": "A", "question": "Create a molecule that satisfies the conditions outlined in the description.\nThe molecule is an indolyl carboxylic acid. It has a role as a metabolite.", "C": "In response, I offer C(C(NC(C(NC(C(CO)N)=O)CS)=O)CC(O)=O)(=O)O"}
{"B": "In response, I offer CC(C(NC(=O)C1C2C(CC(CC)C=1)C(CC2)=O)C(O)=O)CC", "D": "I would conclude that C1(O)C(OC(C(O)C1O)C)OCC1C(C(O)C(O)C(Oc2cc3c([o+]c2-c2cc(c(O)c(O)c2)O)cc(c(c3O)O)O)O1)O", "A": "I would say that c1(N)c(N(C)C(=O)OC)c(nc(-c2c3cccnc3n(Cc3ccccc3F)n2)n1)N", "answer": "D", "question": "Generate a molecule based on the given description.\nThe molecule is a natural product found in Alstroemeria werdermannii with data available.", "C": "My response is C1(=O)CNC(CC(OC(C(NC(=O)C(C)NC(C(NC(=O)C(N1)C(C)C)CC(C)C)=O)C)=O)C(CC(CCCCCC)C)C)=O"}
### Output:
[
    {"task": "multi-choice question answering", "input columns": ["question", "A", "B", "C", "D"], "output columns": ["answer"]},
    {"task": "text-to-molecule generation", "input columns": ["question"], "output columns": ["answer"]},
]

### Input:
1. **Dataset Name**: {{dataset_name}}
2. **Description**: {{description}}
3. **Columns**: {{columns}}
4. **Sample Data**: Here are a few rows from the dataset:
    {{sample_data}}
### Output:
"""

prompt_for_template_generation = """
You are a prompt engineer. You need to design input-output templates to convert the dataframe into natural language for instruction tuning of an LLM.
Given some examples of the dataframe, the name, input columns and output columns for the target task, you need to first verify whether a template is needed to connect the columns, if so, you need to design templates, otherwise, return {"need_template": false}.
The criterion for whether a template is needed is that if the input and output columns can be directly concatenated to form a natural language sentence, then a template is not needed, otherwise, a template is needed.
Task name: {{task_name}}
Input columns: {{input_modality}}
Output columns: {{output_modality}}
Here are some examples of the dataframe:
{{examples}}
The prompts should contain input placeholder and output placeholder (wrapped by {}).
The input and output sentences are split by a \\t character.
The prompts should only contain one \\t character.
The returned prompts should have the same placeholder names.
For example, {"need_template": true, "templates": ["generate a drug for {target_name}\\t{SMILES}", "given a target {target_name}, provide a drug for it.\\t{SMILES}"]}
List at least 10 prompts that can be used for instruction tuning. Only list the results in JSON format: {"need_template": true, "templates": ["Prompt 1", ..., "Prompt 10"]} or {"need_template": false}. Do not add any other characters like \\n or \\t in the JSON format.
"""

prompt_for_extract_code_generation_template = """
Write a python function named 'process_data' that transforms a Hugging Face dataset into the input and output placeholders in a given prompt.
The function should accept the following parameters: a loaded Hugging Face dataset split (a list of dictionaries), a prompt list containing prompts with input and output placeholders wrapped by {}.
For each instance, you should first remove the \\n and \\t of the input and output, randomly select one prompt from the prompt list, fill in the input and output placeholders with the extracted information from the corresponding columns.
The function should return a list of the filled prompts.
Here's the information about the dataset:
1. **Dataset Name**: {{dataset_name}}
2. **Columns**: {{columns}}
3. **Sample Data**: Here are a few rows from the dataset:
    {{sample_data}}
The task is {{task_name}}.
Here's the information about the prompt:
1. **input columns**: {{input_modality}}
2. **output columns**: {{output_modality}}
3. **Prompt List**: {{prompt_list}}
Note that some columns may be in the format of JSON or list, you need to extract the information from the JSON or list and fill in the placeholders.
Some columns may contain unnecessary information or html tags, you need to remove them.
Use tqdm to show the progress of the data processing.
ONLY return the python function, do not include other explanations or code.
"""

prompt_for_extract_code_generation = """
Write a python function named 'process_data' that transforms a Hugging Face dataset into an input-output sentence.
The function should accept the following parameter: a loaded Hugging Face dataset split (a list of dictionaries).
For each instance, you should extract the input and output from the corresponding columns, remove the \\n and \\t of the input and output, and split the input and output by a '\\t' character.
The function should return a list of sentences.
Here's the information about the dataset:
1. **Dataset Name**: {{dataset_name}}
2. **Columns**: {{columns}}
3. **Sample Data**: Here are a few rows from the dataset:
    {{sample_data}}
The task is {{task_name}}.
Here's the information about the prompt:
1. **input columns**: {{input_modality}}
2. **output columns**: {{output_modality}}
Some columns may be in the format of JSON or list, you need to extract the information from the JSON or list.
Some columns may contain unnecessary information or html tags, you need to remove them.
Use tqdm to show the progress of the data processing.
ONLY return the python function, do not include other explanations or code.
"""

prompt_for_instruction_quality_relevance = """
Please review the following instruction-tuning samples.
Each sample is a JSON object with "question" and "answer".

Criterion: every answer should be related to its question and should not be random filler such as "error", "train", "null", or unrelated metadata.

Return only a JSON object:
{"answer": true/false, "reason": "short explanation"}

Data Samples:
{{input}}
Output:
"""

prompt_for_instruction_quality_leakage = """
Please review the following instruction-tuning samples.
Each sample is a JSON object with "question" and "answer".

Criterion: the question should not fully leak the answer. Be conservative: multiple-choice questions that include options are acceptable if the correct option is not directly revealed as the answer.

Return only a JSON object:
{"answer": true/false, "reason": "short explanation"}

Data Samples:
{{input}}
Output:
"""

prompt_for_instruction_quality_language = """
Please review the following instruction-tuning samples.
Each sample is a JSON object with "question" and "answer".

Criterion: the question and answer should be mainly written in English.

Return only a JSON object:
{"answer": true/false, "reason": "short explanation"}

Data Samples:
{{input}}
Output:
"""

prompt_for_instruction_quality_text_only = """
Please review the following instruction-tuning samples.
Each sample is a JSON object with "question" and "answer".

Criterion: the question and answer should be text that an LLM can process directly. They should not contain embedding vectors, long numeric arrays, binary payloads, or other non-text artifacts.

Return only a JSON object:
{"answer": true/false, "reason": "short explanation"}

Data Samples:
{{input}}
Output:
"""

prompt_for_instruction_quality_topic = """
Please review the following instruction-tuning samples.
Each sample is a JSON object with "question" and "answer".

Criterion: the samples should be relevant to this data-curation request:
{{topic}}

Return only a JSON object:
{"answer": true/false, "reason": "short explanation"}

Data Samples:
{{input}}
Output:
"""

prompts_for_instruction_quality_check = [
    prompt_for_instruction_quality_relevance,
    prompt_for_instruction_quality_leakage,
    prompt_for_instruction_quality_language,
    prompt_for_instruction_quality_text_only,
    prompt_for_instruction_quality_topic,
]

prompt_for_rag_process = """
# Overview
You are a coding assistant helping the user build a Retrieval-Augmented Generation (RAG) system.
The original user requirement of this RAG system is:
{{user_query}}

# Task
You need to write Python 3.11 code to process raw data into meaningful content for this RAG system.
Your specific task is to write a python function to process the data into items that can be used by RAG.
An item is a dictionary with the following keys:
- id: a unique identifier for the item
- contents: a string that contains the meaningful contents of the item

# Dos and Don'ts
## Dos
- Analyze the data schema and examples: Examine the columns and data types (e.g., raw strings, JSON strings, integers, floats).
- Identify useful columns: Decide which columns are relevant to the task and determine how to use them effectively.
- Keep the user scenario in mind: Ensure the contents can match end-user queries and answer those queries accurately.
- Handle multi-choice problems properly: Include only the correct answer for multi-choice problems.
- Combine fields to construct id or contents if the required information spans across multiple fields.
- Generate multiple items per item if possible: Write code to create multiple items to cover different aspects of the data, ensuring comprehensive coverage.
- Include necessary context in contents: Ensure all relevant information is captured in the id and contents to provide meaningful output.
- Output the items only: The code should only return items of text extracted from the dataset.
- Ensure the code can handle cases beyond the examples: The full dataset may cover concepts or entities not present in the examples. The only thing you can assume is the JSON schema will be consistent, but the content may vary.
- If you get feedback from the user, also acknowledge the feedback and revise the code accordingly.

## Don'ts
- DO NOT generate incomplete data: Ensure the contents contain all necessary context instead of incomplete or ambiguous information.
- DO NOT miss any columns if they are relevant
- DO NOT add unrelated information in the output: exclude anything beyond the extracted text items, such as errors or additional commentary.
- DO NOT suppress unexpected exceptions. Do Not put the overall logic in a big try-except block.
- DO NOT include any other things except the desired function in or after the code block.
- DO NOT over-fit to the examples: avoid writing code that only works for the provided examples; Avoid using hardcoded values in the code.
- DO NOT truncate any text.
- DO NOT try to use regex or rules to extract details from the data.
- DO NOT use duplicate id.

# Example data:
You can assume all data have the same schema of the example input below.
Here are a list of examples, but your code only needs to process one item each time.

{{sample_data}}

# Output format
Your output should strictly follow this format:
- A step-by-step analysis of the requirements, the user feedback (if it exists), the data schema, and the data itself.
- The expected data items if the code is executed successfully on the example data.
- A data processing plan.
- A python code block that contains a function named `process_data` that takes a string as input and returns a list of dictionaries, where each dictionary contains two keys: the id and the contents.
The python code block should be formulated as
```python
def process_data(data: str) -> list[dict[str, str]]:
    # Your code here
```
- The python code block is mandatory. If you are uncertain, still provide a simple valid implementation that extracts the most relevant fields.
- DO NOT include any other things after the code block.
"""

prompt_for_rag_process_validator = """
# Overview
You are an expert data scientist that helps user validate if both the data processing code and the generated data are correct.
The user's goal is to build a Retrieval-Augmented Generation (RAG) system.
Specifically, this step is to validate:
1. The process_data function code meets the coding requirements
2. The items generated from the raw data are correct
The items will be used for the RAG system where the id is used to match the question and the contents is used to provide information to answer the question.

The original user requirement of this RAG system is:
{{user_query}}

# Task
You will be given the raw data, the process_data function code, and the processed data.
Your task is to validate if both the code and processed data are correct and meet the requirements.

## Code Validation Requirements
Check if the process_data function code follows these requirements:
- Function is named `process_data` and has correct signature: `def process_data(data: str) -> list[dict[str, str]]:`
- The return type is a list of dictionaries with two keys: the 'id' and the 'contents'
- Function takes a string as input and returns a list of dictionaries with two keys: the 'id' and the 'contents'
- Code analyzes data schema properly and identifies useful columns
- Code combines fields to construct id or contents when information spans multiple fields
- Code generates multiple items per item when possible for comprehensive coverage.
- Code does NOT suppress exceptions with broad try-except blocks around main logic
- Code does NOT include hardcoded values that would only work for specific examples
- Code does NOT include regex or rule-based extraction methods
- Code does NOT truncate any text content
- Function contains only the required logic, no extra functions or code after it

## Data Validation Requirements
Check if the processed data meets these requirements:
- The processed data is a JSON array
- Each item in the processed data is a dictionary with two keys: the 'id' and the 'contents'
- The 'id' and 'contents' are both strings
- All the 'id's are unique
- No important information from the raw data is lost in the processed data (missing columns, incomplete or truncated text, etc.). It's OK to omit non-information columns like IDs
- All information in the processed data can be found in the raw data. No extra information is added except for formatting reasons
- It is acceptable for `contents` to be empty if `id` contains all necessary information

# Output format
Your output must follow this EXACT format:

## THINKING:
[Your step-by-step analysis of both the code requirements and data validation rules]

## ERRORS:
[List each error you find on a separate line, starting with "- ". If no errors found, write "None"]

## RESULT:
[Write exactly "CORRECT" if both the code and processed data are correct, or "INCORRECT" if either has errors]

Example:
## THINKING:
I need to check if the process_data function follows the coding requirements and if the processed data follows the data requirements.

## ERRORS:
- Code uses broad try-except block around main logic which suppresses exceptions
- 'id's are not unique: "question1" appears twice in processed data
- Missing information from column 'answer' from the raw data

## RESULT:
INCORRECT
"""

prompt_for_keywords_generation = """
You are a data scientist that helps user find useful datasets from internet such as Hugging Face Hub.

## Task
Given the user requirement, you need to generate a list of keywords related to the user request.
These keywords will be used to search for datasets by name in Hugging Face Hub.

## Requirements
- Coverage is important: we need to recall as many relevant datasets as possible.
- First, include all well-known or widely used datasets that are related to the target domain.
- Second, brainstorm relevant keywords likely to be used in the dataset name.
- No need to use specific dataset names. Instead, find terms (e.g., full names, abbreviations, acronyms, synonyms, etc.) that are commonly used in the target domain. Therefore, the user can find multiple datasets with the same keyword.
- Only include English keywords in lowercase.
- Generate at least 20 diverse keywords.

## Output
In the end of your output, there is a list of keywords in a JSON array format.
```json
["keyword1", "keyword2", ...]
```
Do not include anything else in or after the JSON block.
"""


selection_system_prompt = """
You are a data scientist working in an AI4Science research lab. You are asked to analyze a dataset and judge the quality of the dataset.
The dataset will later be used to run instruction fine-tuning or RAG on scientific LLM models about natural science. Natural science is an inclusive term that includes but is not limited to physics, chemistry, biology, medical science, and so on.

Especially, you are asked to check the following aspects from both metadata and data samples:
1. The correctness and quality of the dataset; only the text columns matter, not other modalities like image or embedding.
2. The relevance of the dataset content and the goal.
3. The language of the dataset should ONLY be English; Otherwise you should give a score of 0.
4. The dataset should not be too large or too small. Especially, you should reject the dataset with > 100M rows.
5. The dataset should be widely used and popular.
6. There is some possibility that the text in the dataset can be used to run instruction fine-tuning or RAG on scientific LLM models about natural science after careful processing.

Your output should contain two parts:
1. A brief report that contains an itemized list checking the above aspects.
2. An overall score of the dataset in [0-9] where 0 is the worst and 9 is the best.
""".strip()
