# modal runner for task 2, unsupervised domain adaptation on pacs
#
# launch it detached, which is what the later pa0 parts used. the container keeps running on modal's side and the
# results land in the volume, so a dropped laptop connection or a closed terminal cannot kill the run:
#     modal run --detach pa1/task2/run_task2_modal.py
#
# the assignment's own hyperparameters are the default. --scale swaps in the larger batch / higher learning rate
# configuration described in SCALED_PARAMETERS below, which is faster on a 96gb card but is a deliberate deviation
# from the values the assignment fixes:
#     modal run --detach pa1/task2/run_task2_modal.py --scale
#
# pull the results back afterwards (executed notebook, csvs, pngs, checkpoints):
#     modal volume get atml /pa1/task2/ ./modal_run
#
# watch a detached run:  modal app logs atml-pa1-task2

from pathlib import Path

import modal

APP_NAME = 'atml-pa1-task2'
VOLUME_NAME = 'atml' #the same volume pa0 used, everything here lives under /pa1 so the two do not mix
VOLUME_PATH = '/mnt/atml'
#the notebook runs with this as its working directory, so its relative './data' and './outputs' paths land on the
#volume. that is what makes the downloaded datasets and every result survive between runs
REMOTE_TASK_DIR = f'{VOLUME_PATH}/pa1/task2'
NOTEBOOK = Path(__file__).with_name('task2.ipynb')

GPU = 'RTX-PRO-6000' #a single fixed gpu type, so the hardware is identical across every run and reportable
CPU_CORES = 8.0
MEMORY_MB = 32768
TIMEOUT_SECONDS = 6 * 60 * 60 #six hours, six runs of up to 30 epochs each

image = (
    modal.Image.debian_slim(python_version='3.11') #the python version pa0's runs used
    #blackwell cards need cuda 12.8 builds, the default pypi wheels are older and would fail to launch a kernel.
    #this call uses the pytorch index, so it is kept separate from the pypi packages below
    .pip_install('torch', 'torchvision', index_url='https://download.pytorch.org/whl/cu128')
    .pip_install(
        'papermill==2.7.0', 'ipykernel', 'numpy', 'pandas', 'matplotlib',
        'seaborn', 'scikit-learn', 'tqdm', 'pillow', 'pyarrow'
    )
    #copy=True bakes the notebook into the image, so a detached run does not depend on the local machine staying up
    .add_local_file(NOTEBOOK, remote_path='/root/task2.ipynb', copy=True)
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)

#empty means the notebook keeps its own constants, which are the ones the assignment fixes
SPEC_PARAMETERS = {}

#papermill injects these over the notebook's constants cell. batches are four times larger and the learning rates
#follow square root scaling (sqrt(4) = 2x) rounded up to 2.5x, so slightly more aggressive than sqrt alone without
#the instability full linear scaling (4x) would risk
SCALED_PARAMETERS = {
    #8 -> 32 per source domain and 24 -> 96 target, keeping the source and target halves equal as the assignment
    #requires. SOURCE_BATCH is listed explicitly because it is derived in the same cell papermill overrides
    'SOURCE_BATCH_PER_DOMAIN': 32,
    'SOURCE_BATCH': 96,
    'TARGET_BATCH': 96,
    'LEARNING_RATE': 2.5e-4,
    'EVAL_BATCH_SIZE': 512, #evaluation is forward only, so this changes speed and nothing else
}


@app.function(gpu=GPU, cpu=CPU_CORES, memory=MEMORY_MB, timeout=TIMEOUT_SECONDS, volumes={VOLUME_PATH: volume})
def run(parameters):
    import papermill

    volume.reload() #picks up anything an earlier task wrote, and the cached datasets
    Path(REMOTE_TASK_DIR).mkdir(parents=True, exist_ok=True)

    executed = f'{REMOTE_TASK_DIR}/task2_executed.ipynb'
    print(f'running task2 with parameters {parameters or "the notebook defaults"}')

    try:
        papermill.execute_notebook(
            '/root/task2.ipynb', executed,
            parameters=parameters,
            cwd=REMOTE_TASK_DIR, #relative paths inside the notebook resolve here, on the volume
            kernel_name='python3',
            progress_bar=False, #the logs are captured, a progress bar would only add noise to them
            log_output=True, #every cell's stdout goes to the modal logs, so a detached run is still followable
        )
    finally:
        #committed even when a cell raises, so the partially executed notebook and whatever results exist are kept
        volume.commit()
        print(f'results committed to {REMOTE_TASK_DIR}')


@app.local_entrypoint()
def main(scale: bool = False):
    run.remote(SCALED_PARAMETERS if scale else SPEC_PARAMETERS)
