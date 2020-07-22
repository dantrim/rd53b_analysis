#!/bin/bash

venv_dir_name="env"
python_cmd="python"
required_python_version_major=3
required_python_version_minor=6

function print_usage {
    echo "---------------------------------------------------------"
	echo " setup for RD53B analysis"
    echo ""
    echo " This script initializes a python virtual environment"
    echo " in which to run the analysis infrastructure."
    echo " The virtual environment will be named \"${venv_dir_name}\","
    echo " and once set-up can be exited at anytime by calling"
    echo " 'deactivate' at the command-line."
    echo ""
    echo " If the ${venv_dir_name}/ directory is not found, this script"
    echo " installs all of the necessary packages needed to"
    echo " run the testbench. See the setup.py"
    echo " file for a complete list of packages that will be"
    echo " installed in the virtual environment"
    echo ""
    echo " Options:"
    echo "  -h|--help       print this help message"
    echo ""
    echo " Example usage:"
    echo "  $ source setup_env.sh [OPTIONS]"
    echo ""
    echo " Note:"
    echo "  - This script must be run every time you wish to run"
    echo "    the testbench infrastructure (any previous virtual"
    echo "    environment must first be deactivated)."
    echo ""
    echo " Requires:"
    echo "  - python3 version >= ${required_python_version_major}.${required_python_version_minor}"
    echo "---------------------------------------------------------"
}

function python3_available {

    if ! command -v python >/dev/null 2>&1; then
        echo "ERROR python not installed"
        return 1
    fi

    ##
    ## check if the default command satisfies the python requirements
    ##
    $(python -c "import sys; sys.exit({True:0,False:1}[sys.version_info[1]>=${required_python_version_minor}]);")    
    if [ $? -eq 1 ]; then
        if ! command -v python3 >/dev/null 2>&1; then
            echo "ERROR python3 not installed"
            return 1
        fi
        $(python3 -c "import sys; sys.exit({True:0,False:1}[sys.version_info[1]>=${required_python_version_minor}]);")
        if [ $? -eq 1 ]; then
            echo "ERROR You must have python version >=3.6, you have $(python3 -V) (checked with: \"python3 -V\")"
            return 1
        fi
        python_cmd=python3
    fi
    return 0

    #if ! command -v python3 >/dev/null 2>&1; then
    #    echo "ERROR python3 not installed"
    #    return 1
    #fi

    #$(python3 -c "import sys; sys.exit({True:0,False:1}[sys.version_info[1]>=${required_python_version_minor}]);")
    #if [ $? -eq 1 ]; then
    #    echo "ERROR You must have python version >=3.6, you have $(python3 -V) (checked with: \"python3 -V\")"
    #    return 1
    #fi
    #return 0
}

function update_pip {
    ${python_cmd} -m pip install --upgrade --no-cache-dir pip setuptools wheel 2>&1 >/dev/null
    status=$?
    if [[ ! $status -eq 0 ]]; then
        echo "ERROR Problem in updating pip within the virutal environment"
        return 1
    fi
    return 0
}

function activate_venv {
    source ${venv_dir_name}/bin/activate
    if [ $? -eq 1 ]; then
        echo "ERROR Could not activate virtual environment \"${venv_dir_name}\""
        return 1
    fi

    if ! update_pip; then
        return 1
    fi
    return 0
}

function pre_commit_setup {
    if ! command -v pre-commit -V >/dev/null 2>&1; then
        echo "ERROR pre-commit not installed"
        return 1
    fi

    ##
    ## put pre-commit hooks in .git
    ##
    pre-commit install 2>&1 >/dev/null
    return 0
}

function main {

    while test $# -gt 0
    do
        case $1 in 
            -h)
                print_usage
                return 0
                ;;
            --help)
                print_usage
                return 0
                ;;
            *)
                echo "ERROR Invalid argument: $1"
                return 1
                ;;
        esac
        shift
    done

    ##
    ## enforce python 3
    ##
    if ! python3_available; then
        return 1
    fi

    ##
    ## setup
    ##
    if [ -d ${venv_dir_name} ]; then
         if ! activate_venv ; then
            return 1
        fi

        if ! pre_commit_setup; then
            return 1
        fi

    else
        ${python_cmd} -m venv ${venv_dir_name}
        if [ ! -d ${venv_dir_name} ]; then
            echo "ERROR Problem setting up virtual environment \"${venv_dir_name}\""
            return 1
        else 

            if ! activate_venv ; then
                return 1
            fi
            if ! python -m pip install --quiet -e . ; then
                echo "ERROR There was a problem in installing the packages"
                deactivate >/dev/null 2>&1 
                return 1
            fi

            ##
            ## setup pre-commit
            ##
            if ! pre_commit_setup; then
                return 1
            fi

            echo "Installation successful"
        fi
    fi

    echo "Virtual environment \"${venv_dir_name}\" has been activated. Type 'deactivate' to exit."
}

#__________________________________
main $*
