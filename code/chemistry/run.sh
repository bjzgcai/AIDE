# collect instruction tuning data
python main.py

# data filtering
cd src/instagent
python dedup.py
cd ../..
pip install open-dataflow[vllm]
mkdir dataflow
cd dataflow
dataflow init
cp ../src/instagent/dataflow.py gpu_pipelines
cd gpu_pipelines
python dataflow.py