prompt_for_usage_brain_storming = """
I have a dataset, and I would like you to brainstorm how it can be used in scientific tasks for instruction tuning of LLM.
The task should be compatible with all the samples in the dataset and can be organized as input-output pairs.
Your output should be a JSON array where each item describes a potential task for instruction tuning. Each item in the JSON array should include:
- **'task'**: A short description of the proposed task.
- **'input columns'**: A list of columns that will be used as input for the task.
- **'output columns'**: A list of columns that will be used as output for the task.
The column names MUST be chosen from the given columns.
You should try your best to keep the original task for the dataset (e.g., muti-choice question answering for AI4Chem/ChemBench4K), and you can also add new tasks if you think they are valuable.
Try to avoid some naive tasks like "count the number of rows" or "count the number of columns".
ONLY return the result in the following JSON format:
[
    {"task": "task1", "input columns": ["column1", "column2", ...], "output columns": ["column1", "column2", ...]},
    {"task": "task2", "input columns": ["column1", "column2", ...], "output columns": ["column1", "column2", ...]}
]
If you think there is no valid tasks (e.g., the dataset only contains one column, etc.), please return an empty JSON array [].
Here are some examples:
## Example 1
### Input:
1. **Dataset Name**: clinicalnlplab/BC5CDR_chemical_train
2. **Description**:
3. **Columns**: unproccessed, processed
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
    {"task": "multi-choice question answering", "input columns": ["question", "A", "B", "C", "D"], "output columns": ["answer", "column['answer']"]},
    {"task": "text-to-molecule generation", "input columns": ["question"], "output columns": ["column['answer']"]},
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
List at least 10 prompts that can be used for instruction tuning. Only List the results in JSON format: {"need_template": true, templates: [Prompt 1 ... Prompt 10]} or {"need_template": false}, do not add any other characters like \\n or \\t in the JSON format.
"""

prompt_for_extract_code_generation_template = """
Write a python function named 'process_data' that transforms a list of rows into the input and output placeholders in a given prompt. Each row is a dictionary representing a single instance in the dataset.
The function should accept the following parameters: a loaded Hugging Face dataset split (a list of dictionary), a prompt list containing prompts with input and output placeholders wrapped by {}.
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
Write a python function named 'process_data' that transforms a Hugging Face dataset into a input-output sentence.
The function should accept the following parameter: a loaded Hugging Face dataset split (a list of dictionary).
For each instance, you should extract in the input and output from the corresponding columns, remove the \\n and \\t of the input and output, and split the input and output by a '\\t' character.
The function should return a list of the sentence.
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

prompt_for_data_post_processing = """
Given a sentence containing input and output information, you need to post-process it as follows:
1. If the input and output information are not separated by a '\t' character, you need to split them by the '\t' character.
2. If the sentence contains more than one '\t' character, you should only keep the '\t' separating the input and output information.
3. If the sentence contains '\n', you should remove them.
4. If the sentence contains SMILES of molecules, you should wrap the SMILES with '<mol>' and '</mol>'.
5. If the sentence contains DNA sequences (i.e, sequence formed by A/T/C/G), you should wrap the DNA sequences with '<dna>' and '</dna>'.
6. If the sentence contains RNA sequences (i.e, sequence formed by A/U/C/G), you should wrap the RNA sequences with '<rna>' and '</rna>'.
7. If the sentence contains protein FASTA sequences, you should wrap the protein sequences with '<protein>' and '</protein>'.
8. If the sentence contains the composition of material (e.g., 'Al2NO3'), you should wrap the composition with '<material>' and '</material>', and flatten the composition (e.g., <material>Al Al N O O O</material).
ONLY return the post-processed sentence.
Here are some examples:
- Input: The molecule is C1=CC=CC=C1\tThe molecule is benzene.
- Output: The molecule is <mol>C1=CC=CC=C1</mol>\tThe molecule is benzene.

- Input: The DNA sequence is ATCG\tThe RNA sequence is AUCG.
- Output: The DNA sequence is <dna>ATCG</dna>\tThe RNA sequence is <rna>AUCG</rna>.

- Input: The protein sequence is ARNDCEQGHILKMFPSTWYV\tThe protein sequence is ARNDCEQGHILKMFPSTWYV.
- Output: The protein sequence is <protein>ARNDCEQGHILKMFPSTWYV</protein>\tThe protein sequence is <protein>ARNDCEQGHILKMFPSTWYV</protein>.

- Input: The composition is Al2O3\tThe material contains Al and O.
- Output: The composition is <material>Al Al O O O</material>\tThe material contains Al and O.

- Input: Hi,\nI am a sentence.\\tI am another sentence.
- Output: Hi, I am a sentence.\tI am another sentence.

- Input: {{input}}
- Output:
"""

prompt_for_quality_1 = """
Please review the following data samples and determine if each of the data samples is:
the answer is related to the question, not some random words such as "error", "train", "null".
your final answer should be true if **ALL** the data are consistent with the above criteria, otherwise output false.
output a json file with the following format:
{"answer": true/false, "reason": "the reason why you think the answer is true or false"}
Only return the json file.
Only give an answer and reason, do not give them sample by sample.
Data Samples:
{{input}}
Output:
"""
prompt_for_quality_2 = """
Please review the following data samples and determine if each of the data samples is:
the question should not self-asked and self-answered. 
this criteria is not strict. ONLY when the "question" column directly repeat ALL the "answer" section, it is considered a self-asked and self-answered question. Otherwise, it should not be treated as such. 
An example of self-asked and self-answered question is "{"question": What is the capital of France? The capital of France is Paris. "answer": Paris}".
An example of NOT self-asked and self-answered question is "{"question": Is the capital of France Paris? "answer": Is the capital of France Paris? Yes, it is Paris}". 
Additionally, multi-choice questions without explicitly indicating the correct option should also not be treated as self-asked and self-answered. 
For example, "{"question": What is the capital of France? A. Paris B. London C. Berlin D. Madrid "answer": A. Paris}" is not a self-asked and self-answered question.
{"question": What is the capital of France? A. Paris B. London C. Berlin D. Madrid, the answer is A", answer": A. Paris} is a self-asked and self-answered question.
your answer should be true if **ALL** the data not self-asked and self-answered, otherwise output false.
output a json file with the following format:
{"answer": true/false, "reason": "the reason why you think the answer is true or false"}
Only return the json file.
Only give one answer and reason, do not give them sample by sample.
Data Samples:
{{input}}
Output:
"""

prompt_for_quality_3 = """
Please review the following data samples and determine if each of the data samples is:
the question and answer are mainly written in English.
your final answer should be true if **ALL** the data are consistent with the above criteria, otherwise output false.
output a json file with the following format:
{"answer": true/false, "reason": "the reason why you think the answer is true or false"}
The answer should be true if **ALL** the data are consistent with the above criteria, otherwise output false.
Only return the json file.
Only give an answer and reason, do not give them sample by sample.
Data Samples:
{{input}}
Output:
"""
prompt_for_quality_4 = """
Please review the following data samples and determine if each of the data samples is:
the question and answer do not contain embedding vector, which is a long sequence of numbers. Note that sequence of text is not considered as embedding vector.
your answer should be true if **ALL** the data are consistent with the above criteria, otherwise output false.
output a json file with the following format:
{"answer": true/false, "reason": "the reason why you think the answer is true or false"}
Only return the json file.
Only give an answer and reason, do not give them sample by sample.
Data Samples:
{{input}}
Output:
""" 
prompt_for_quality_5 = """
Please review the following data samples and determine if each of the data samples is:
the question and answer is kind of related to {{topic}}.
your final answer should be true if **ALL** the data are consistent with the above criteria, otherwise output false.
output a json file with the following format:
{"answer": true/false, "reason": "the reason why you think the answer is true or false"}
Only return the json file.
Only give an answer and reason, do not give them sample by sample.
Data Samples:
{{input}}
Output:
"""
prompts_for_quality_check = [
    prompt_for_quality_1,
    prompt_for_quality_2,
    prompt_for_quality_3,
    prompt_for_quality_4,
    prompt_for_quality_5,
]
prompt_for_rag_process = """
You are an assistant tasked with writing Python code to process structured data into chunks for a Retrieval-Augmented Generation (RAG) system.

### Task
Your task is to write a python function to process the data into chunks that can be used by RAG.
A chunk consists of two parts: an index and a value, both of them are text. When the user gives a query, the index will be used to match the queries. And the values of top matched indexes will be given to the user to answer the query.

### Requirements
1. The data comes from internet so make sure you check the data carefully.
2. Keep the user scenario in mind. Make sure the index can be used to match the end-user queries, and the value can be used to answer the queries.
3. If the data is a multi-choice problem, the value should only include the correct answer, not all answers.
4. Feel free to more complex processing like combining the fields to construct indexes or queries if the information spans across different fields.
5. If any field contains a json string (i.e., nested json), please make sure you extract the information from the json string and use it in the index or value.
6. For each item in the dataset, you are encouraged to write code that can generate multiple index-value pairs to cover different aspects of the data.
7. Ensure your code is robust and can handle corner cases like missing values.
8. If only one column is valid, you can use the column name as the index and empty string as the value.
9. Ensure all necessary context is included in the index and value.

You can assume all data have the same schema of the example input below.
Here are a list of examples, but your code only needs to process one item each time.
### Example data:
{{sample_data}}

### OUTPUT
You need first think step by step.
Then followed by a python code block (i.e., ```python).
The python only contains one function process_data, i.e, def process_data(data: str) -> list[tuple[str, str]].
"""

prompt_for_quality_score = """
You are an expert in evaluating the quality of instruction tuning data samples.
Your task is to review the following data samples and assign a quality score based on the following criteria:
1. **Relevance**: The question and answer should be relevant to the {{topic}}.
2. **Clarity**: The question should be well-defined, clear and unambiguous, and the answer should be concise and directly address the question.
3. **Language**: The question and answer should be written in English, and should not contain any embedding vector or random words.
4. **Factual Accuracy**: The answer should be factually correct and not contain any false information.
5. **Coherence**: The question and answer should be coherent and logically connected.
Your final score should be a number between 0 and 10, where 0 means the data sample is of very low quality and 10 means the data sample is of very high quality.
Output a JSON object with the following format:
{"score": 0-10, "reason": "the reason why you think the score is X"}
Only return one JSON dict representing the overall quality score for all the data samples, not a score for each sample.
Data Samples:
{{input}}
Output:
"""