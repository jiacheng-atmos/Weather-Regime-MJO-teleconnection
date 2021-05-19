from mpl_toolkits.basemap import Basemap as basemap
import numpy as np
from numpy import dtype
import netCDF4 as nc
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, interp
from sklearn.cluster import KMeans
import sklearn
import scipy
from scipy import stats,signal
from eofs.standard import Eof


### Pre-processing: the sample data we choose is hgt500 from 2000.1.1 to 2019.12.31
# The desired data format is ny x nt x nlat x nlon; here the actual range if from 2000.12.1 to 2019.3.31
# There are 19 winters totally. Each winter consists of 121 days. We negelect leap days.
# lat range: 70N-10N (from north to south); lon range: 210-320 (from west to east, covering the whole North American Continent)
nt=121; nlon=111; nlat=61; ny=37; noef=4; ncluster=4; num_montecarlo=1000; tele_length=16
path='/data/keeling/a/jye18/c/GEFS-reforecast/MJO/reanalyasis/'
ps = xr.open_dataset(path+'hgt500_1982_2019_DJFM_70N10N_210_320_daily_era5_1d.nc')
hgt500_era50=ps['z'].values/9.80665
hgt500_era5=np.zeros((ny,nt,nlat,nlon))
a=90 # remove 1983.1.1-1983.3.31; to keep only 19 winters
for iyear in np.arange(1983,2020):
    if np.mod(iyear,4)==0: 
        hgt500_era5[iyear-1983]=hgt500_era50[a:a+nt]
        a=a+122
    else:  # delete leap days
        hgt500_era5[iyear-1983]=hgt500_era50[a:a+nt]
        a=a+121

lat_at=ps['lat'].values; lon_at=ps['lon'].values
del hgt500_era50

cos = lat_at * np.pi / 180.0
way = np.cos(cos)
weightf = np.repeat(way[:,np.newaxis],len(lon_at),axis=1)   # add weighting function (because of the latitude)
atemp_era5 = signal.detrend(hgt500_era5,axis=0)  # # linearly detrend 500 hPa geopotential height data
atemp_era5_pre=np.zeros((nt*ny,nlat,nlon))
for iy in np.arange(ny):
    atemp_era5_pre[iy*nt:iy*nt+nt]=atemp_era5[iy]*weightf[None,:,:]

### we did not using n-day moving average as some other studies do partly because the original goal for us (Jiacheng & Zhuo)
###   is to evaluate GEFS v12 reforecasts where the reforecast length is usually 16 days which made it impossible
###   to apply n-day moving average or other time filtering methods. 4 EOFs may already filter some noisy signals
### To be consistent with other reseachers, one can add n-day moving code above. 
# EOF analysis
solver = Eof(atemp_era5_pre, center=True)
pcs = solver.pcs() 
mid_eig=solver.eigenvalues() 
mid_eofs=solver.eofs()
eofs = solver.eofs()

### Print explained variance when using 4 EOFs
#var_explained_era5= np.cumsum(mid_eig)/np.sum(np.sum(mid_eig))
#print(var_explained_era5[3])
#0.5316330300316366

reconstruction_era5 = solver.reconstructedField(noef) #Using 4 leading EOFs to reconstruct hgt500 field 

### The Kmeans method needs a 2-D data format: number of days x horizontal fields
atemp_era5_post=np.zeros((ny*nt,nlat*nlon))
for i in np.arange(ny*nt):
    atemp_era5_post[i]=(reconstruction_era5[i]).flatten()

### For other regimes, one probably should include red-noise test (e.g., Vigaud et al. 2018)
### Define 4 clusters (consistent with previous studies). Different random state won't change the results very much
km_standard = KMeans(n_clusters=ncluster,random_state=0)
km_standard = km_standard.fit(atemp_era5_post)
label_era5 = km_standard.labels_ # (total number of days: ny x nday = 37 x 121 = 4477 for sample data)

mid_label_yearday0 = np.reshape(label_era5,[ny,nt]) # Reshape label into 2-D array ny x nt
#mid_finalcluster = km_standard.cluster_centers_    # cluster centers
#mid_finalcluster = np.reshape(mid_finalcluster,[4,nlat,nlon]) # # reshape mid_finalcluster
label_4pattern_era5=np.zeros((ncluster,ny))  # number of days for each regime for each year
for iy in np.arange(ny):
    for i in np.arange(ncluster):
        label_4pattern_era5[i,iy]=np.sum(mid_label_yearday0[iy]==i)
perc_era5=np.zeros(4) # percentage of each regime to the total number of days
for ii in np.arange(4):
    perc_era5[ii] = np.sum(label_era5==ii)/len(label_era5)
    
atemp_era5_posts=reconstruction_era5.reshape(ny,nt,nlat,nlon)
cluster_era5=np.zeros((ncluster,nlat,nlon)) # cluster centers: equivalent to mid_finalcluster
for imode in np.arange(ncluster):
    midi=mid_label_yearday0==imode
    cluster_era5[imode]=np.nanmean(atemp_era5_posts[midi],axis=0)
    
# one-student significance test;
itest=np.zeros((4,nlat,nlon))>0
for imode in np.arange(ncluster):
    mid=cluster_era5[imode]
    midi=mid_label_yearday0==imode
    itest[imode]=scipy.stats.ttest_1samp(atemp_era5[midi],0,axis=0).pvalue>0.05

# plot figure for each regime
#mode_name=['(a) Arctic High (NAO-)','(b) West Coast Ridge','(c)Pacific Trough','(d) Pacific Ridge'] # for sample data
mode_name=['(a)','(b)','(c)','(d)']
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(10,6))
x,y = np.meshgrid(lon_at,lat_at)
imode=0
for axx in axes.T.flat:

    mmid=cluster_era5[imode].copy()
    mmid[itest[imode]]=np.nan
    m = basemap(projection="cyl",llcrnrlat=10,urcrnrlat=70.,\
                llcrnrlon=210,urcrnrlon=320,ax=axx,resolution='c')
    m.drawcoastlines(linewidth=2, color="k")
    m.drawparallels(np.arange(30,90,30.),labels=[1,0,0,0],fontsize=15)
    m.drawmeridians(np.arange(210,340,30.),labels=[0,0,0,1],fontsize=15)
    X,Y = m(x,y)
    levels=np.arange(-150,150.1,15)
    im0 = m.contourf(X,Y,mmid,levels,cmap = plt.get_cmap('RdBu_r'),extend='both')
    im1 = m.contour(X,Y,mmid,levels,colors='k')
    axx.set_title(mode_name[imode],fontsize=18,loc='left')
    imode=imode+1
plt.suptitle('North American Weather Regimes(ERA5)',fontsize=20,y=1.05)
cbar_ax = fig.add_axes([0.15, -0.05, 0.7, 0.03]) # left, bottom height ,length, width
fig.colorbar(im0,orientation='horizontal',cax=cbar_ax)
fig.tight_layout()
print('Normal End!') 

# The teleconnection part (MJO - weather regime) is the only section that is hard to avoid hard-coding
path='/data/keeling/a/jye18/c/GEFS-reforecast/MJO/reanalyasis/rmm.74toRealtime.txt'
data1 = pd.read_csv(path,sep='\s+',header=1)
RMM_au_phase=data1.iloc[2771:16650,5].values-1 # ranging from 1983.1.1 to 2019.12.31; eight different MJO phase
RMM_au_mag=data1.iloc[2771:16650,6].values   # RMM amplitude


# assign MJO active days as True so as to helping us to do the following teleconnection work
midi_djfm=~np.ones((8, ny,nt), dtype=bool)
a=0
for iyear in np.arange(1982,2019):
    if np.mod(iyear,4)==0:
        mid=RMM_au_phase[a+335:a+335+nt]
        for iday in np.arange(nt):
            if RMM_au_mag[a+335+iday]>=1:
                midi_djfm[RMM_au_phase[a+335+iday],iyear-1982,iday]=True
        a=a+366
    else:
        mid=RMM_au_phase[a+334:a+334+nt]
        for iday in np.arange(nt):
            if RMM_au_mag[a+335+iday]>=1:
                midi_djfm[RMM_au_phase[a+334+iday],iyear-1982,iday]=True
        a=a+365

# calculate transition probability matrix (ncluster x ncluster)
wg_transition_table_era5=np.zeros((ny, nt-1,ncluster,ncluster))+np.nan
mid1=mid_label_yearday0[:,1:]
mid2=mid_label_yearday0[:,:-1]
for iy in np.arange(ny):
    for iday in np.arange(nt-1):
        wg_transition_table_era5[iy,iday,int(mid2[iy,iday]),int(mid1[iy,iday])]=1

mid2=np.nansum(wg_transition_table_era5,axis=(0,1))
transition_table_era5=mid2/(np.sum(mid2,axis=1))[:,None]
# Calculate the number of regime days lagged by different MJO leading days
total_num_each_regime_era5=np.zeros((tele_length,8,ncluster))
for ifd in np.arange(tele_length): 
    for iphase in np.arange(8):
        midi=midi_djfm[iphase,:,:nt-tele_length]
        mid1=mid_label_yearday0[:,ifd:nt-tele_length+ifd][midi]
        for imode in np.arange(ncluster):
            total_num_each_regime_era5[ifd,iphase,imode]=np.sum(mid1==imode)
# Apply monte carlo test: Assign random cluster number at each year's 1st Dec, then using transition matrix to 
#   achieve the whole winter cluster number array. Repeat this process for nyear then repeat the above 
#   steps for 1000 times.
tran_era5=transition_table_era5
markov_initiation=np.zeros((num_montecarlo,ny,nt))
for imar in np.arange(num_montecarlo):
    for iy in np.arange(ny):
        markov_initiation[imar,iy,0]=np.random.randint(0,ncluster) # assign random cluster number on 1 Dec
        for iday in np.arange(1,nt):
            mid=markov_initiation[imar,iy,iday-1]
            markov_initiation[imar,iy,iday]= np.random.choice(np.arange(0, ncluster), p=tran_era5[int(mid)]) # markov process
# Calculate the number of regime days lagged by different MJO leading days from artificial dataset generated monte-carlo simulation
total_num_each_regime_era5_markov=np.zeros((num_montecarlo,tele_length,8,ncluster))
for imar in np.arange(num_montecarlo):
    for ifd in np.arange(tele_length): 
        for iphase in np.arange(8):
            midi=midi_djfm[iphase,:,:nt-tele_length]
            mid1=markov_initiation[imar,:,ifd:nt-tele_length+ifd][midi]
            for imode in np.arange(ncluster):
                total_num_each_regime_era5_markov[imar,ifd,iphase,imode]=np.sum(mid1==imode)
# significance test: here 90%
markov_era5_sigtest=np.zeros((tele_length,8,ncluster))<0
for ifd in np.arange(tele_length):
    for iphase in np.arange(8):
        for imode in np.arange(ncluster):
            mid1=np.percentile(total_num_each_regime_era5_markov[:,ifd,iphase,imode], 95) # 95% 
            mid2=np.percentile(total_num_each_regime_era5_markov[:,ifd,iphase,imode], 5)  # 5%
            mid_bound=total_num_each_regime_era5[ifd,iphase,imode]
            if mid_bound>=mid1 or mid_bound<=mid2:
                markov_era5_sigtest[ifd,iphase,imode]=True
                
fig, axes = plt.subplots(nrows=1, ncols=4,figsize=(15,3))
#mtype1=['(a) AH 1982-2019','(b) WCR','(c) PT','(d) PR'] # for the sample data
mtype1=['(a)','(b)','(c)','(d)']
n=0
level_mean=np.arange(0,26)
for axx in axes.T.flat:
    imode=n
    mid1=np.nanmean(total_num_each_regime_era5_markov,axis=0)[:,:,imode]
    mid2=total_num_each_regime_era5[:,:,imode]
    mmid=(mid2-mid1)/mid1*100 # anomalous percentage of weather regime occurrence freq lagged by different MJO leading days 
    im0=axx.pcolor(np.arange(9),np.arange(17),mmid,cmap = plt.get_cmap('RdBu_r'),vmin=-70,vmax=70)
    mmid[~markov_era5_sigtest[:,:,imode]]=np.nan
    axx.pcolor(np.arange(9),np.arange(17),mmid, hatch='.', alpha=0.)
    axx.set_title(mtype1[imode],loc='left',fontsize=18)
    axx.set_xticks(np.array([1,2,3,4,5,6,7,8])-0.5)  ; 
    axx.set_xticklabels(['P1','P2','P3','P4','P5','P6','P7','P8'])
    axx.set_yticks(np.array([1,6,11,16])-0.5)  ; 
    axx.set_yticklabels(['0','5','10','15'])
    axx.tick_params(labelsize=12)
    axx.set_ylabel('MJO Lead Days',fontsize=15)
    n=n+1
cbar_ax = fig.add_axes([0.15, -0.05, 0.7, 0.03]) # left, bottom height ,length, width
fig.colorbar(im0,orientation='horizontal',cax=cbar_ax)
fig.tight_layout()

