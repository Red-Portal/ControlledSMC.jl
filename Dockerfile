FROM julia:1.11-bookworm

ENV USER smc
ENV USER_HOME_DIR /home/${USER}
ENV JULIA_DEPOT_PATH ${USER_HOME_DIR}/.julia

RUN useradd -m -d ${USER_HOME_DIR} ${USER}

# Copy source files
ADD *.toml ${USER_HOME_DIR}/
ADD scripts ${USER_HOME_DIR}/scripts

RUN julia -e "cd(\"${USER_HOME_DIR}\"); using Pkg; Pkg.add(\"https://github.com/Red-Portal/ControlledSMC.jl/tree/stepsize_adaptation.git\"); Pkg.activate(\"scripts\"); Pkg.develop([\"ControlledSMC\"])"

RUN julia -e "cd(\"${USER_HOME_DIR}\"); using Pkg; Pkg.activate(\"scripts\"); Pkg.update(); Pkg.precompile(); Pkg.status(); println(pwd());"
 
RUN chmod -R a+rwX ${USER_HOME_DIR}

USER ${USER}

# configure the script entry point
WORKDIR ${USER_HOME_DIR}

ENTRYPOINT ["julia", "-e", "using Distributed, SysInfo; addprocs(SysInfo.ncores() > 80 ? div(SysInfo.ncores(), 2) : 40); @everywhere using Pkg; @everywhere Pkg.activate(\"scripts\"); @everywhere include(\"scripts/experiments.jl\"); main()"]