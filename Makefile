##############################################
#
#  Makefile to install Python in virtualenv
#  with all dependencies for FlyTracker.
#
##############################################
# MARCHE PAS ENCORE :)
##############################################

VENV_LIB = venv/lib/python3.5
VENV_CV2 = $(VENV_LIB)/cv2.so

# Find cv2 library for the global Python installation.
GLOBAL_CV2 := $(shell python3 -c 'import cv2; print(cv2)' | awk '{print $$4}' | sed s:"['>]":"":g)



# Copy global cv2 library file into the virtual environment.
$(VENV_CV2): $(GLOBAL_CV2) venv
	echo $(GLOBAL_CV2) 
	cp $(GLOBAL_CV2) $@

venv: requirements.txt
	test -d venv || virtualenv -p python3 venv
	. venv/bin/activate && pip3 install -r requirements.txt

test: $(VENV_CV2)
	. venv/bin/activate && python -c 'import cv2; print(cv2)'

track: venv
	test -d venv || virtualenv -p python3 venv
	. venv/bin/activate && python FlyTracking.0.01.py -i /media/neberle/6CEC5E62EC5E271C/Backup_Linux/fly_tracker/dataset/fly_tracker_1280_960_3.h264

clean:
	rm -rf venv


