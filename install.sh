#!/bin/sh
# install script for ana_cont library
# some parts are copied from w2dynamics install script.

BASE_DIR=$(pwd)
python setup.py build_ext --inplace
rm -r build

if [ -f $HOME/.profile ]; then
    PROFILE_FILE=$HOME/.profile
elif [ -f $HOME/.bash_profile ]; then
    PROFILE_FILE=$HOME/.bash_profile
else
    echo "No bash startup file found!"
    echo "Please set PROFILE_FILE in the install script by hand!"
fi


if ! grep "# BEGIN ana_cont added" $PROFILE_FILE >/dev/null; then
    cat <<EOF >>$PROFILE_FILE
# BEGIN ana_cont added
# Adding ana_cont to your Pythonpath in:
source $BASE_DIR/.profile
# END ana_cont added
EOF
fi


export PYTHONPATH=$PYTHONPATH:$BASE_DIR


echo "export PYTHONPATH=\$PYTHONPATH:$BASE_DIR" > .profile
source .profile
