import streamlit as st
import joblib
import numpy as np
from os import path
#import pandas as pd
import matplotlib.pyplot as plt


from os import path
import sys
sys.path.insert(0, path.abspath('../'))


def parse_model_parameter_file(parfile):
    pardict = {}
    f = open(parfile, 'r')
    for line in f:
        par = line.split("#")[0]
        if par != "":
            par = par.split(":")
            key = par[0]
            val = [ival.strip() for ival in par[1].split(",")]
            for i in range(1, 3):
                val[i] = float(val[i])
            pardict.update({key: val})
    return pardict


@st.cache(allow_output_mutation=True)
def loadEmulator():
    emuList = ["Emulator_AuAu200_dNdy.joblib",
               "Emulator_AuAu200_pTvn.joblib",
               "Emulator_AuAu200_PHOBOSdNdeta.joblib",
               "Emulator_AuAu200_PHOBOSv2eta.joblib",
               "Emulator_AuAu19p6_dNdy.joblib",
               "Emulator_AuAu19p6_pTvn.joblib",
               "Emulator_AuAu19p6_PHOBOSdNdeta.joblib",
               "Emulator_AuAu7p7_dNdy.joblib",
               "Emulator_AuAu7p7_pTvn.joblib",
              ]
    emus = []
    for emu_i in emuList:
        emus.append(joblib.load(path.join("Emulators", emu_i)))
    return emus


def main():
    emuList = loadEmulator()
    exp_data = np.loadtxt("exp_data_RHIC_STAR+PHOBOS_wv3.dat")

    # The title of the page
    st.title('3D-Glauber + MUSIC + UrQMD Model for RHIC BES energies')

    st.write("This is an interactive web page that emulates "
             + "particle production and their anisotropic flow "
             + "as functions of rapidity "
             + "using the 3D-Glauber+MUSIC+UrQMD model.")
    st.write("This work is based on [link]"
             + "(xxx)")
    st.write("One can adjust the model parameters on the left sidebar.")
    st.write("The colored bands in the figure show the emulator estimations "
             + "with their uncertainties. "
             + "The compared experimental data are from the STAR and "
             + "PHOBOS Collaborations")

    # Define model parameters in the sidebar
    modelParamFile = "3DMCGlauber.txt"
    paraDict = parse_model_parameter_file(modelParamFile)
    st.sidebar.header('Model Parameters:')
    params = []     # record the model parameter values
    for ikey in paraDict.keys():
        parMin = paraDict[ikey][1]
        parMax = paraDict[ikey][2]
        parInit = (parMax + parMin)/2.
        parVal = st.sidebar.slider(label=paraDict[ikey][0],
                                   min_value=parMin, max_value=parMax,
                                   value=parInit,
                                   step=(parMax - parMin)/1000.,
                                   format='%f')
        params.append(parVal)
    params = np.array([params,])

    # make model prediction using the emulator
    param0 = np.copy(params)
    param1 = np.copy(params)
    param1[0, 15] = params[0, 16]; param1[0, 17] = params[0, 16]
    param2 = np.copy(params)
    param2[0, 15] = params[0, 17]; param2[0, 16] = params[0, 17]
    paramList = [param0, param0, param0, param0, param1, param1, param1,
                 param2, param2]
    pred = np.array([])
    predErr = np.array([])
    for i, emu_i in enumerate(emuList):
        modelPred, modelPredCov = emu_i.predict(paramList[i], return_cov=True)
        pred = np.concatenate((pred, modelPred[0, :]))
        predErr = np.concatenate((predErr,
                                  np.sqrt(np.diagonal(modelPredCov[0, :, :]))))

    cenList = np.array([2.5, 7.5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
    dNcencut = 7
    vncencut = 6

    Nrap = 46
    rapArr = np.linspace(-4.5, 4.5, Nrap)
    offset = 68

    # make plot
    # pid dN/dy @ 200 GeV
    offset = 0
    fig1 = plt.figure()
    for ipart in range(4):
        id0 = offset + ipart*dNcencut
        id1 = id0 + dNcencut
        plt.errorbar(cenList[:dNcencut],
                     exp_data[id0:id1, 0], exp_data[id0:id1, 1],
                     color='k', marker='o', linestyle='', label="STAR")
        plt.fill_between(cenList[:dNcencut],
                         pred[id0:id1] + predErr[id0:id1],
                         pred[id0:id1] - predErr[id0:id1], alpha=0.5)
    plt.xlabel(r"Centrality (%)")
    plt.ylabel(r"$dN/dy$")
    plt.yscale('log')

    # pid mean pT @ 200 GeV
    offset += 4*dNcencut
    fig2 = plt.figure()
    for ipart in range(4):
        id0 = offset + ipart*dNcencut
        id1 = id0 + dNcencut
        plt.errorbar(cenList[:dNcencut],
                     exp_data[id0:id1, 0], exp_data[id0:id1, 1],
                     color='k', marker='o', linestyle='', label="STAR")
        plt.fill_between(cenList[:dNcencut],
                         pred[id0:id1] + predErr[id0:id1],
                         pred[id0:id1] - predErr[id0:id1], alpha=0.5)
    plt.xlabel(r"Centrality (%)")
    plt.ylabel(r"$\langle p_T \rangle$ (GeV)")
    plt.ylim([0, 1.5])

    # charged hadron vn @ 200 GeV
    offset += 4*dNcencut
    fig3 = plt.figure()
    for ipart in range(2):
        id0 = offset + ipart*vncencut
        id1 = id0 + vncencut
        plt.errorbar(cenList[:vncencut],
                     exp_data[id0:id1, 0], exp_data[id0:id1, 1],
                     color='k', marker='o', linestyle='', label="STAR")
        plt.fill_between(cenList[:vncencut],
                         pred[id0:id1] + predErr[id0:id1],
                         pred[id0:id1] - predErr[id0:id1], alpha=0.5)
    plt.xlabel(r"Centrality (%)")
    plt.ylabel(r"$v_n$")
    plt.ylim([0, 0.12])

    col1, col2, col3 = st.columns(3)
    col1.pyplot(fig1)
    col2.pyplot(fig2)
    col3.pyplot(fig3)

    # dNch/deta @ 200 GeV
    cenLabels = ["0-5%", "5-12%", "12.5-23.5%", "23.5-33.5%", "33.5-43.5%"]
    offset += 2*vncencut
    fig4 = plt.figure()
    for ipart in range(5):
        id0 = offset + ipart*Nrap
        id1 = id0 + Nrap
        plt.errorbar(rapArr, exp_data[id0:id1, 0], exp_data[id0:id1, 1],
                     color='k', marker='o', linestyle='',
                     label=cenLabels[ipart])
        plt.fill_between(rapArr, pred[id0:id1] + predErr[id0:id1],
                         pred[id0:id1] - predErr[id0:id1], alpha=0.5)
    plt.legend()
    plt.xlim([-5, 5])
    plt.ylim([0, 900])
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$dN^\mathrm{ch}/d\eta$")

    # v2(eta) @ 200 GeV
    offset += 5*Nrap
    v2rapArr = np.array([-4.90, -4.16, -3.41, -2.70, -2.25, -1.75,
                         -1.26, -0.76, -0.3, 0.3, 0.76, 1.26, 1.75,
                         2.25, 2.70, 3.41, 4.16, 4.94])
    v2Nrap = v2rapArr.size - 4
    fig5 = plt.figure()
    id0 = offset
    id1 = id0 + v2Nrap
    plt.errorbar(v2rapArr[2:-2], exp_data[id0:id1, 0], exp_data[id0:id1, 1],
                 color='k', marker='o', linestyle='', label="PHOBOS")
    plt.fill_between(v2rapArr[2:-2], pred[id0:id1] + predErr[id0:id1],
                     pred[id0:id1] - predErr[id0:id1], alpha=0.5)
    plt.legend()
    plt.xlim([-5, 5])
    plt.ylim([0, 0.08])
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$v_2(\eta)$")

    # pid dN/dy @ 19.6 GeV
    offset += v2Nrap
    fig1 = plt.figure()
    for ipart in range(3):
        id0 = offset + ipart*dNcencut
        id1 = id0 + dNcencut
        plt.errorbar(cenList[:dNcencut],
                     exp_data[id0:id1, 0], exp_data[id0:id1, 1],
                     color='k', marker='o', linestyle='', label="STAR")
        plt.fill_between(cenList[:dNcencut],
                         pred[id0:id1] + predErr[id0:id1],
                         pred[id0:id1] - predErr[id0:id1], alpha=0.5)
    plt.xlabel(r"Centrality (%)")
    plt.ylabel(r"$dN/dy$")
    plt.yscale('log')

    # pid mean pT @ 19.6 GeV
    offset += 3*dNcencut
    fig2 = plt.figure()
    for ipart in range(4):
        id0 = offset + ipart*dNcencut
        id1 = id0 + dNcencut
        plt.errorbar(cenList[:dNcencut],
                     exp_data[id0:id1, 0], exp_data[id0:id1, 1],
                     color='k', marker='o', linestyle='', label="STAR")
        plt.fill_between(cenList[:dNcencut],
                         pred[id0:id1] + predErr[id0:id1],
                         pred[id0:id1] - predErr[id0:id1], alpha=0.5)
    plt.xlabel(r"Centrality (%)")
    plt.ylabel(r"$\langle p_T \rangle$ (GeV)")
    plt.ylim([0, 1.5])

    # charged hadron vn @ 19.6 GeV
    offset += 4*dNcencut
    fig3 = plt.figure()
    for ipart in range(2):
        id0 = offset + ipart*vncencut
        id1 = id0 + vncencut
        plt.errorbar(cenList[:vncencut],
                     exp_data[id0:id1, 0], exp_data[id0:id1, 1],
                     color='k', marker='o', linestyle='', label="STAR")
        plt.fill_between(cenList[:vncencut],
                         pred[id0:id1] + predErr[id0:id1],
                         pred[id0:id1] - predErr[id0:id1], alpha=0.5)
    plt.xlabel(r"Centrality (%)")
    plt.ylabel(r"$v_n$")
    plt.ylim([0, 0.12])

    col1, col2, col3 = st.columns(3)
    col1.pyplot(fig1)
    col2.pyplot(fig2)
    col3.pyplot(fig3)

    # dNch/deta @ 19.6 GeV
    offset += 2*vncencut
    fig6 = plt.figure()
    for ipart in range(5):
        id0 = offset + ipart*Nrap
        id1 = id0 + Nrap
        plt.errorbar(rapArr, exp_data[id0:id1, 0], exp_data[id0:id1, 1],
                     color='k', marker='o', linestyle='',
                     label=cenLabels[ipart])
        plt.fill_between(rapArr, pred[id0:id1] + predErr[id0:id1],
                         pred[id0:id1] - predErr[id0:id1], alpha=0.5)
    plt.legend()
    plt.xlim([-5, 5])
    plt.ylim([0, 450])
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$dN^\mathrm{ch}/d\eta$")

    # pid dN/dy @ 7.7 GeV
    offset += 5*Nrap
    fig1 = plt.figure()
    for ipart in range(3):
        id0 = offset + ipart*dNcencut
        id1 = id0 + dNcencut
        plt.errorbar(cenList[:dNcencut],
                     exp_data[id0:id1, 0], exp_data[id0:id1, 1],
                     color='k', marker='o', linestyle='', label="STAR")
        plt.fill_between(cenList[:dNcencut],
                         pred[id0:id1] + predErr[id0:id1],
                         pred[id0:id1] - predErr[id0:id1], alpha=0.5)
    plt.xlabel(r"Centrality (%)")
    plt.ylabel(r"$dN/dy$")
    plt.yscale('log')

    # pid mean pT @ 7.7 GeV
    offset += 3*dNcencut
    fig2 = plt.figure()
    for ipart in range(4):
        id0 = offset + ipart*dNcencut
        id1 = id0 + dNcencut
        plt.errorbar(cenList[:dNcencut],
                     exp_data[id0:id1, 0], exp_data[id0:id1, 1],
                     color='k', marker='o', linestyle='', label="STAR")
        plt.fill_between(cenList[:dNcencut],
                         pred[id0:id1] + predErr[id0:id1],
                         pred[id0:id1] - predErr[id0:id1], alpha=0.5)
    plt.xlabel(r"Centrality (%)")
    plt.ylabel(r"$\langle p_T \rangle$ (GeV)")
    plt.ylim([0, 1.5])

    # charged hadron vn @ 7.7 GeV
    offset += 4*dNcencut
    fig3 = plt.figure()
    for ipart in range(2):
        id0 = offset + ipart*vncencut
        id1 = id0 + vncencut
        plt.errorbar(cenList[:vncencut],
                     exp_data[id0:id1, 0], exp_data[id0:id1, 1],
                     color='k', marker='o', linestyle='', label="STAR")
        plt.fill_between(cenList[:vncencut],
                         pred[id0:id1] + predErr[id0:id1],
                         pred[id0:id1] - predErr[id0:id1], alpha=0.5)
    plt.xlabel(r"Centrality (%)")
    plt.ylabel(r"$v_n$")
    plt.ylim([0, 0.12])

    col1, col2, col3 = st.columns(3)
    col1.pyplot(fig1)
    col2.pyplot(fig2)
    col3.pyplot(fig3)

    col1, col2, col3 = st.columns(3)
    col1.pyplot(fig4)
    col2.pyplot(fig5)
    col3.pyplot(fig6)

if __name__ == '__main__':
    main()
