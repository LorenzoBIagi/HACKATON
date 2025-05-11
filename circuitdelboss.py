{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "W = []\n",
    "nk = np.array([1,2,4,8,8,4,2])\n",
    "\n",
    "for k in range(len(cores)):\n",
    "        \n",
    "        j, i, n_k = cores[k].shape\n",
    "        #print(cores[k].shape)\n",
    "        cores[k] = np.reshape(cores[k], (j * i, n_k)) \n",
    "        #print(cores[k].shape)       \n",
    "        U, S, V = np.linalg.svd(cores[k])\n",
    "        R = np.diag(S) @ V\n",
    "        if k != len(cores)-1:\n",
    "            cores[k+1] = np.einsum('j r,r i k->j i k', R, cores[k+1])\n",
    "        W.append(U)\n",
    "\n",
    "\n",
    "for k, U in enumerate(W):\n",
    "    print(U.shape)\n",
    "    start = k + 1\n",
    "    mn    = min(start, np.log2(nk[k])) \n",
    "    diff  = int(start - mn)\n",
    "    print(f\"This unitary acts from qubit {int(diff)} to qubit {start}\")\n",
    "\n",
    "    #for i in range(0,diff+1):\n",
    "    #     U = np.kron(np.eye(2),U)\n",
    "\n",
    "    #for j in range(start+1,len(W)):\n",
    "    #     U = np.kron(U,np.eye(2))\n",
    "\n",
    "    print(U.shape)"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
