default:
    just --list


# installs OpenCV using vcpkg (clone vcpgk if not already cloned)
vcpkg_install_opencv:
    if [ ! -d vcpkg ]; then \
      git clone https://github.com/microsoft/vcpkg.git ; \
      cd vcpkg ; \
      ./bootstrap-vcpkg.sh ; \
      ./vcpkg install opencv  ; \
    else \
      echo "vcpkg already cloned" ; \
    fi


install_requirements:
    pip install -r requirements.txt

