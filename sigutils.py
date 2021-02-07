  
"""
    These are the utilities to be used for extraction of different 
    attributes from NFH , Sensor pulse and GCS 90 data from gunlink
    Seg-D files.

    List of Utilities:

    1.Apply a filter to data 
    2.Detect peaks on data
    3.Extract bubble period , peak onset time and peak onset amplitude from the data
    4.Find xcorr cofficient from the data
    5.Find spectral deveation from the data 

"""


import numpy as np
from scipy.signal import butter, lfilter, convolve2d
from scipy import signal 





def butterworth_highpass(data ,freq,  fs, order=5 , padlen=100):
    """
    Function to Apply butterworth highpass filter to a 
    single trace of data function is divided in two parts


    Args:
        data (1D-ndarray): Single trace to apply the filter on 
        frequency (float): cutoff frequency for the filter
        fs (int): Sampling frequency of the measuring system.
        order (int, optional): Order of the butterworth filter. Defaults to 5.
        padlen (int, optional): padlength for filter application . Defaults to 5.
    Returns:
        filtered (1D-ndarray): filtered trace 

    """
    #Calculate nyquist frequency
    nyq = 0.5 * fs
    freq = freq / nyq

    """
    Design the filter and generate filter cofficients 

    b: ndarray (Numerator polynomial of the IIR filter)
    a: ndarray (Denominator polynomial of the IIR filter)

    """
    b, a = butter(order, freq, btype='high')
    filtered = signal.filtfilt(b, a, data, padtype='odd', padlen=101)

    return filtered 
    




def butterworth_lowpass(data ,freq,  fs, order=5 , padlen=100):
    """
    Function to Apply butterworth lowpass filter to a 
    single trace of data function is divided in two parts


    Args:
        data (1D-ndarray): Single trace to apply the filter on 
        frequency (float): cutoff frequency for the filter
        fs (int): Sampling frequency of the measuring system.
        order (int, optional): Order of the butterworth filter. Defaults to 5.
        padlen (int, optional): padlength for filter application . Defaults to 5.
    Returns:
        filtered (1D-ndarray): filtered trace 

    """
    #Calculate nyquist frequency
    nyq = 0.5 * fs
    freq = freq / nyq

    """
    Design the filter and generate filter cofficients 

    b: ndarray (Numerator polynomial of the IIR filter)
    a: ndarray (Denominator polynomial of the IIR filter)

    """
    b, a = butter(order, freq, btype='high')
    filtered = signal.filtfilt(b, a, data, padtype='odd', padlen=101)

    return filtered 
    




def butterworth_bandpass(data ,freq_low, freq_high, fs, order=5 , padlen=100):
    """
    Function to Apply butterworth lowpass filter to a 
    single trace of data function is divided in two parts


    Args:
        data (1D-ndarray): Single trace to apply the filter on 
        freq_low (float): low cutoff frequency for the filter
        freq_high (float): high cutoff frequency for the filter
        fs (int): Sampling frequency of the measuring system.
        order (int, optional): Order of the butterworth filter. Defaults to 5.
        padlen (int, optional): padlength for filter application . Defaults to 5.
    Returns:
        filtered (1D-ndarray): filtered trace 

    """
    #Calculate nyquist frequency
    nyq = 0.5 * fs
    freq = freq / nyq

    """
    Design the filter and generate filter cofficients 

    b: ndarray (Numerator polynomial of the IIR filter)
    a: ndarray (Denominator polynomial of the IIR filter)

    """
    b, a = butter(order, [freq_low, freq_high], btype='band')
    filtered = signal.filtfilt(b, a, data, padtype='odd', padlen=101)

    return filtered 




def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False, show=False, ax=None):
    """
    Function to detect peaks on a given signal

    Args:
        x (1D-ndarray): Input trace for peak detection
        mph (int, optional): Minimum Peak height, Defaults to None.
        mpd (int, optional): Minimum distance between peaks Defaults to 1.
        
    Returns:
        list: A list of all the indexes where the peaks were detected
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        #print(len(x))
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind



def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """
    Helper function to QC and check the peaks by plotting 
    them on the given signal.

    Is called by the show=True argument in the detect_peaks 
    function.

    """
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
            
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Trace Data', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                    % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()






def xcorr(x, y, normed=True, detrend=False, maxlags=10):
    """
    Function to claculate the cross-corelation between two 
    signals of equal length

    Returns the coefficients when normed=True
    Returns inner products when normed=False
    Usage: lags, c = xcorr(x,y,maxlags=len(x)-1)

    Args:
        x (1D-ndarray): 1st signal
        y (1D-ndarray): 2nd signal
        normed (bool, optional): Returns the coefficients when normed=True,  Defaults to True.
        detrend (bool, optional): If you need to remove the linear trend from the signal/Series, Defaults to False.
        maxlags (int, optional): Offset between the signals,  Defaults to 10.

    Raises:
        ValueError: If the length of two signals is not equal
        

    Returns:
        list: lags 
        float: cross co-relation cofficients
    """
   
    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')

    if detrend:
        import matplotlib.mlab as mlab
        x = mlab.detrend_mean(np.asarray(x))  # can set your preferences here
        y = mlab.detrend_mean(np.asarray(y))

    c = np.correlate(x, y, mode='full')

    if normed:
        n = np.sqrt(np.dot(x, x) * np.dot(y, y))  # this is the transformation function
        c = np.true_divide(c, n)

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maglags must be None or strictly '
                         'positive < %d' % Nx)

    lags = np.arange(-maxlags, maxlags + 1)
    c = c[Nx - 1 - maxlags:Nx + maxlags]
    
    return lags, c



def bubble_attributes(singleTrace, mpd=80 , threshold=0, mph=0.2):
    """
    Funtion to return the values of 
    -Bubble period 
    -Peak On-set time 
    -Peak On-set amplitude 

    * Uses the detect_peaks() function

    Args:
        singleTrace (1D-ndarray): Single trace from which bubble attributes
        need to be extracted.

    Returns:
        bp (float): bubble period
        poa (float): peak onset amplitude
        pot (float): peak onset time 
    """
    singleTrace = -singleTrace
    peak_onset_amplitude = (singleTrace.max())
    index = detect_peaks(singleTrace, mpd=mpd, show=True, threshold=threshold, mph=mph)
        
    time_peaks = (index/2)
    bubble_period = time_peaks[1] - time_peaks[0]
    peak_onset_time = time_peaks[0]

    return peak_onset_amplitude, peak_onset_time, bubble_period






