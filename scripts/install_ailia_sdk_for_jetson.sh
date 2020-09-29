python3 -c "import site; print (site.getsitepackages())"

export SITEPACKAGES=/usr/local/lib/python3.6/dist-packages
export AILIA_SDK=./ailia_1_24_0

sudo cp -r ${AILIA_SDK}/python/ailia ${SITEPACKAGES}
sudo cp ${AILIA_SDK}/library/jetson/libailia.so ${SITEPACKAGES}/ailia
sudo cp ${AILIA_SDK}/library/jetson/libailia_pose_estimate.so ${SITEPACKAGES}/ailia
sudo cp ${AILIA_SDK}/library/jetson/libailia_cuda-8.so ${SITEPACKAGES}/ailia
sudo cp ${AILIA_SDK}/library/jetson/libailia_cuda-7.so ${SITEPACKAGES}/ailia

sudo chmod 757 ${SITEPACKAGES}/ailia
