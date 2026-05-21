from dataflow.operators.filter import (
    WordNumberFilter,
    SuperfilteringFilter,
    DeitaQualityFilter,
    InstagFilter
)
from dataflow.utils.storage import FileStorage


class SFTTextFilter_GPUPipeline():
    
    def __init__(self):
        
        self.storage = FileStorage(
            first_entry_file_name="/tos-mlp-zgci/liuzequn/instagent/train_chemistry.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        
        self.model_cache_dir = './dataflow_cache'
        self.word_number_filter_step1 = WordNumberFilter(
            min_words=20,
            max_words=1000
        )
        self.super_filtering_filter_step2 = SuperfilteringFilter(
            min_score=0.5,
            max_score=1.0,
            model_cache_dir=self.model_cache_dir
        )
    def forward(self):
        
        self.word_number_filter_step1.run(
            storage=self.storage.step(),
            input_key="output",
        )
        
        self.super_filtering_filter_step2.run(
            storage=self.storage.step(),
            input_instruction_key='instruction',
            input_input_key=None,
            input_output_key='output'
        )

if __name__ == "__main__":
    # This is the entry point for the pipeline
    pipeline = SFTTextFilter_GPUPipeline()
    pipeline.forward()