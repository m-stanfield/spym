
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(fileNames=[]):
    #fileNames is a list 
    df = pd.DataFrame()
    d = []
    if len(fileNames)>0:
        for name in fileNames:
            print(name)
            tempdf = pd.read_pickle(name)
            tempdf['idx'] = tempdf.index
            tempdf['fileName'] = name
            d.append(tempdf)
    else:
        print('No Files Loaded')
            
    df = pd.concat(d,ignore_index=True)
    return df

def split_data(df,EPP_loc='feature',normalizeI=True,zeroPercent=None,phase_coef=False):
    Iw_i = np.array(df['Iw_i'].tolist())
    Iw_f = np.array(df['Iw_f'].tolist())

    if phase_coef == False:
        phase = np.array(df['Phase_i'].tolist())
    else:
        phase = np.array(df['phase_coef'].tolist())

    EPP = np.array(df['EPP'].values.reshape((-1,1)))
    if normalizeI:
        Iw_i = (Iw_i.T/np.sum(Iw_i,axis=1)).T
        Iw_f = (Iw_f.T/np.sum(Iw_f,axis=1)).T

    if not(zeroPercent == None):
        freq = np.linspace(0,Iw_i.shape[1]-1,Iw_i.shape[1])
    spec = (Iw_i.T/np.max(Iw_i,axis=1)).T
    zeroIndex = ((0.0*spec+freq).T<np.argmax(spec>zeroPercent,axis=1)).T
    phase[zeroIndex] = 0.0
    zeroIndex = ((0.0*spec+freq[::-1]).T<np.argmax((spec>zeroPercent)[:,::-1],axis=1)).T
    phase[zeroIndex] = 0.0
    phase[zeroIndex] = 0.0

    if EPP_loc == 'feature':
        features = list(np.hstack((Iw_i,Iw_f,EPP)))
        targets = list(phase)

    elif EPP_loc == 'target':
        features = list(np.hstack((Iw_i,Iw_f)))
        targets = list(np.hstack((phase,EPP)))
    else:
        print('Error in EPP_loc. Requires either \'feature\' or \'target\'')
    d = {'features':features,'targets':targets,'idx':list(df['idx']),'fileName':list(df['fileName'])}
    #print(d['features'].shape)
    dF = pd.DataFrame(d)
   # for ii in range(len(Iw_i)):
    #    d = {'features':[features[ii]],'targets':[targets[ii]],'idx':[df['idx'][ii]],'fileName':[df['fileName'][ii]]}
    #    dF = dF.append(d,ignore_index=True)
    
    return dF


def plot_data(df,num_plots=1,use_sample=False):
    if use_sample == True:
        tmpdf = df.sample(num_plots)
    else:
        tmpdf = df.iloc[0:num_plots]
    freq = tmpdf['freq'].iloc[0]
    iFreq  = np.linspace(-2000,2000,2**14) + np.mean(freq)
    for ii in range(num_plots):
        i_Iwi = np.interp(iFreq,freq,tmpdf['Iw_i'].iloc[ii],left=0.0,right=0.0)
        i_Iwf = np.interp(iFreq,freq,tmpdf['Iw_f'].iloc[ii],left=0.0,right=0.0)
        i_phase = np.interp(iFreq,freq,tmpdf['Phase_i'].iloc[ii])
        
        Efi = np.sqrt(i_Iwi)*np.exp(1j*i_phase)
        Eti = np.fft.fftshift(np.fft.fft(Efi))
        Eti = np.roll(Eti,int(Eti.shape[0]*0.5-np.argmax(np.abs(Eti)**2)))
        It = np.abs(Eti)**2
        Efi = np.fft.ifft(np.fft.ifftshift(Eti))
        i_Iwi = np.abs(Efi)**2
        i_phase = np.unwrap(np.angle(Efi))
        i_phase -= i_phase[int(i_phase.shape[0]*0.5)]
        time = 1000*np.fft.fftshift(np.fft.fftfreq(iFreq.shape[0],iFreq[1]-iFreq[0]))
        

        fig, ax = plt.subplots(ncols=2)
        ax[0].plot(time,It,'b')
        ax[0].set_xlim([-100,100])
        ax[1].plot(iFreq,i_Iwi/np.max(i_Iwi),'b')
        ax[1].plot(iFreq,i_Iwf/np.max(i_Iwf),'k')
        axP = ax[1].twinx()
        axP.plot(iFreq,i_phase,'r--')
        axP.set_xlim([300,450])
        axP.set_ylim([-10,10])



def pulse_builder(df,features,targets,interp_range=2000.0,num_bins=2**12,cutoff_percent=0.005): 
    freq = df['freq'].iloc[0]
    iFreq  = np.linspace(-interp_range/2,interp_range/2,num_bins) + np.mean(freq)
    
    i_Iw = np.array([np.interp(iFreq, freq, features[ii,0:(freq.shape[0])]) for ii in range(features.shape[0])])
    i_phase = np.array([np.interp(iFreq, freq, targets[ii]) for ii in range(targets.shape[0])])

    Iw_f = np.array([np.interp(iFreq, freq, features[ii,(freq.shape[0]):-1]) for ii in range(features.shape[0])])
    
    EPP = np.array([features[ii,-1] for ii in range(features.shape[0])])
#    i_Iw = np.interp(iFreq,freq,features[:,0:(freq.shape[0])],axis=1)
#    i_phase = np.interp(iFreq,freq,targets,axis=1)
    time = 1000*np.fft.fftshift(np.fft.fftfreq(iFreq.shape[0],iFreq[1]-iFreq[0]))

    Ew = np.multiply(np.sqrt(i_Iw),np.exp(1j*i_phase))
    Et = np.fft.fftshift(np.fft.fft(Ew,axis=1),axes=(1,))
    #for ii in range(Et.shape[0]):
    #    Et[ii] = np.roll(Et[ii],int(Et[ii].shape[0]*0.5-np.argmax(np.abs(Et[ii])**2)))
    It = np.abs(Et)**2
    #Ew = np.fft.ifft(np.fft.ifftshift(Et),axis=1)
    #Iw = np.abs(Ew)**2
    #phase = np.unwrap(np.angle(Ew),axis=1)
    phase = i_phase
    phase -= phase[:,int(0.5*phase.shape[1])].reshape(-1,1)
    t_phase = np.angle(Et) 
    pulse_dict = {'time':time,'It':It,'t_phase':t_phase,'iFreq':iFreq,'i_Iw':i_Iw,'phase':phase,'Iw_f':Iw_f,'EPP':EPP}
    return pulse_dict

def result_compare(df,features,t_targets,p_targets,interp_range=2000.0,number_bins=2**12,cutoff_percent=0.005):
    freq = df['freq'].iloc[0]
    _,t_It,_,_,t_Iw,t_phase,_,_ = pulse_builder(df,features,t_targets,interp_range=interp_range,num_bins=number_bins,cutoff_percent=cutoff_percent)
    _,p_It,_,_,p_Iw,p_phase,_,_ = pulse_builder(df,features,p_targets,interp_range=interp_range,num_bins=number_bins,cutoff_percent=cutoff_percent)
    
    phase_error = np.mean(np.abs(t_phase-p_phase)**2*t_Iw,axis=1)/np.mean(t_Iw,axis=1)
    time_error = np.mean(np.abs(t_It-p_It)**2,axis=1)

    results = {'phase_error':phase_error,'time_error':time_error}
    return results
    
def _test():
    print('Data Splitter Loaded Properly')
if __name__ == '__main__':
    import glob

    plt.close('all')
    files = glob.glob(r'test_1.pkl')   #list of all files you want to load here
    df = load_data(files)
    head = df.head()
    features, targets = split_data(head)
    a = pulse_builder(df,features,targets)
    print(result_compare(df,features,targets,targets)['phase_error'])
    #plot_data(df,num_plots=10)
    
