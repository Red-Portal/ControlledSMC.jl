FROM julia:1.11-bookworm

ENV USER smc
ENV USER_HOME_DIR /home/${USER}
ENV JULIA_DEPOT_PATH ${USER_HOME_DIR}/.julia

RUN useradd -m -d ${USER_HOME_DIR} ${USER}

# Copy source files
ADD *.toml ${USER_HOME_DIR}/
ADD scripts ${USER_HOME_DIR}/scripts

RUN julia -e "using Pkg; Pkg.add([\"Distributed\", \"SysInfo\"])"

RUN julia -e "cd(\"${USER_HOME_DIR}\"); using Pkg; Pkg.activate(\"scripts\"); Pkg.add(url=\"https://github.com/Red-Portal/ControlledSMC.jl.git\", rev=\"stepsize_adaptation\"); Pkg.update(); Pkg.precompile(); Pkg.status(); println(pwd());"

RUN chmod -R a+rwX ${USER_HOME_DIR}

RUN apt-get update && apt-get install -y \
    build-essential \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    autoconf \
    git \
    wget

USER ${USER}

RUN git clone --recurse-submodules https://github.com/roualdes/bridgestan.git ${USER_HOME_DIR}/.bridgestan
RUN mv ${USER_HOME_DIR}/bridgestan ${USER_HOME_DIR}/.bridgestan
ENV BRIDGESTAN=${USER_HOME_DIR}/.bridgestan

RUN julia -e "cd(\"${USER_HOME_DIR}\"); using Pkg; Pkg.activate(\"scripts\"); include(\"scripts/build_posteriordb.jl\"); main()"

# configure the script entry point
WORKDIR ${USER_HOME_DIR}

ENTRYPOINT ["julia", "-e", "using Distributed, SysInfo; addprocs(2); @everywhere using Pkg; @everywhere Pkg.activate(\"scripts\"); @everywhere include(\"scripts/experiments.jl\"); main()"]