 sudo alsa force-reload


result=$?
if [ ${result} == 0 ]; then
  ./bin/app
fi
