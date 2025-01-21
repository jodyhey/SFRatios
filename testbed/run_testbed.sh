#!/bin/bash
# set SF_Ratios.py  to non debugging mode,  i.e. set
# Get current date for results directory
DATE=$(date +"%Y-%m-%d")
RESULTS_DIR="$DATE"

# Create the results directory if it doesn't exist
if [ ! -d "$RESULTS_DIR" ]; then
    mkdir "$RESULTS_DIR"
    echo "Created directory: $RESULTS_DIR"
else
    echo "Directory already exists: $RESULTS_DIR"
fi

# Function to run command and capture error
run_command() {
    local cmd="$1"
    local job_name="$2"
    local error_file="${RESULTS_DIR}/${job_name}_error.txt"
    
    $cmd 2> >(tee "$error_file" >&2) || {
        echo "Error occurred in job: $job_name" >> "$error_file"
        return 1
    }
    
    # If no error occurred, remove the error file
    rm "$error_file"
}

# Array of commands with different models and options

commands=(
  "python ../SF_Ratios.py -a testbed_fixed2Ns_data.txt -f foldit -d fixed2Ns -i 1 -r $RESULTS_DIR -p test1"
  "python ../SF_Ratios.py -a testbed_fixed2Ns_data.txt -f unfolded -d fixed2Ns -i 1 -c 1 -r $RESULTS_DIR -p test2"
  "python ../SF_Ratios.py -a testbed_fixed2Ns_data.txt -f foldit -d fixed2Ns -i 1 -y -r $RESULTS_DIR -p test3"
  "python ../SF_Ratios.py -a testbed_fixed2Ns_data.txt -f unfolded -d fixed2Ns -i 2 -g -r $RESULTS_DIR -p test4"
  "python ../SF_Ratios.py -a testbed_fixed2Ns_data.txt -f foldit -d fixed2Ns -g -u -r $RESULTS_DIR -p test5"
  "python ../SF_Ratios.py -a testbed_normal_data.txt -f foldit -d normal -i 1  -r $RESULTS_DIR -p test6"
  "python ../SF_Ratios.py -a testbed_normal_pm0_data.txt -f foldit -d normal -i 1 -y  -r $RESULTS_DIR -p test7"
  "python ../SF_Ratios.py -a testbed_normal_pm0_data.txt -f foldit -d normal -i 1 -z  -r $RESULTS_DIR -p test8"
  "python ../SF_Ratios.py -a testbed_lognormal_data.txt -f unfolded -d lognormal -i 1 -m 1 -r $RESULTS_DIR -p test9"
  "python ../SF_Ratios.py -a testbed_lognormal_data.txt -f foldit -d lognormal -i 1 -t -r $RESULTS_DIR -p test10"
  "python ../SF_Ratios.py -a testbed_lognormal_pm0_data.txt -f unfolded -d lognormal -i 1 -m 1 -y -r $RESULTS_DIR -p test11"
  "python ../SF_Ratios.py -a testbed_lognormal_pm0_data.txt -f unfolded -d lognormal -i 1 -m 1 -z -r $RESULTS_DIR -p test12"
  "python ../SF_Ratios.py -a testbed_fixed2Ns_pm0_data.txt -f unfolded -d fixed2Ns -i 1 -m 1 -z -r $RESULTS_DIR -p test13"
  "python ../SF_Ratios.py -a testbed_lognormal_data.txt -f foldit -d lognormal -i 1 -m 1 -c 1 -y -r $RESULTS_DIR -p test14"
  "python ../SF_Ratios.py -a testbed_gamma_data.txt -f foldit -d gamma -i 1 -m 1 -r $RESULTS_DIR -p test15"
  "python ../SF_Ratios.py -a testbed_gamma_data.txt -f unfolded -d gamma -i 1 -t  -r $RESULTS_DIR -p test16"
  "python ../SF_Ratios.py -a testbed_gamma_pm_data.txt -f foldit -d gamma -i 1 -m 0 -y -r $RESULTS_DIR -p test17"
  "python ../SF_Ratios.py -a testbed_fixed2Ns_data.txt -f unfolded -d fixed2Ns -i 1 -M 25 -u -g -r $RESULTS_DIR -p test18"
)

# Run all commands in parallel
for cmd in "${commands[@]}"; do
  job_name=$(echo "$cmd" | grep -oP -- '-p \K\w+')
  echo "Running: $cmd"
  run_command "$cmd" "$job_name" &
done

# Wait for all background jobs to finish
wait

echo "All jobs completed."