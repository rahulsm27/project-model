#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

if [["${IS_PROD_ENV}" == "true"]]; then 
    echo "production service"
else 
    /start-tracking-server.sh &
    tail -F anything
fi


#set -o errexit: This option makes the script exit immediately if any command it runs exits with a non-zero status (i.e., it fails). It is also known as set -e. This helps in making scripts more robust by stopping execution on errors.

#set -o pipefail: This option extends the behavior of the -o errexit option. Normally, a pipeline returns the exit status of the last (rightmost) command to exit with a non-zero status, or zero if all commands exit successfully. With pipefail enabled, the pipeline's return status is the value of the last (rightmost) command to exit with a non-zero status, or zero if all commands exit successfully. This is useful when dealing with pipelines to ensure that the entire pipeline fails if any part of it fails.

#set -o nounset: This option treats unset variables as errors and causes the script to exit. If any variable is accessed before being set or assigned a value, the script will terminate. It is also known as set -u and helps catch potential programming errors related to undefined variables.

#By combining these options, you can create more robust and reliable shell scripts by ensuring early termination on errors, better handling of pipeline failures, and prevention of accidental use of undefined variables