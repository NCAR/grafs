import math
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

################# Extra-terrestrial Irradiance ########################################
def calc_TOA(lat, lon, alt, day, month, year, ToD):
    # Solar constant for the mean distance between the Earth and sun #######
    sol_const = 1367.8
    ########################################################################
    # GEO Parameters
    #lat = 23
    #lon = 11
    #alt = 0
    ########################################################################
    # Time and Offsets
    #day             =               22
    #month           =               06
    #year            =               2011
    #ToD             =               12
    tz_off_deg      =               lon #originally was written as (0 + lon)
    dst_off         =               8
    ########################################################################
    # Atmospheric Parameters
    # air temperature
    atm_temp        =               25.0
    # relative humidity
    atm_hum         =               20.0    # Default
    # turbidity coefficient - 0 < tc < 1.0 - where tc = 1.0 for clean air
    # and tc < 0.5 for extremely turbid, dusty or polluted air
    atm_tc          =               0.8     # Default
    ########################################################################
    ##  MAIN
    ########################################################################
    # get Julian Day (Day of Year)
    if calendar.isleap(year):
        # Leap year, 366 days
        lMonth = [0,31,60,91,121,152,182,213,244,274,305,335,366]
    else:
        # Normal year, 365 days
        lMonth = [0,31,59,90,120,151,181,212,243,273,304,334,365]
    DoY = lMonth[month-1] + day
    ## print "--------------------------------------------------------------"
    ## print "%d.%d.%d | %d | %f | " % (day, month, year, DoY, ToD, )
    ## print "--------------------------------------------------------------"
    ## print "Solar Constant                               : %s" % sol_const
    ## print "Atmospheric turbidity coefficient            : %s" % atm_tc
    ## print "--------------------------------------------------------------"
    # inverse relative distance factor for distance between Earth and Sun ##
    sun_rel_dist_f  = 1.0/(1.0-9.464e-4*math.sin(DoY)-                      \
                    + 0.01671*math.cos(DoY)-                                \
                    + 1.489e-4*math.cos(2.0*DoY)-2.917e-5*math.sin(3.0*DoY)-\
                    + 3.438e-4*math.cos(4.0*DoY))**2
    ## print "Inverse relative distance factor             : %s" % sun_rel_dist_f
    # solar declination ####################################################
    sun_decl        = (math.asin(0.39785*(math.sin(((278.97+(0.9856*DoY))   \
                    + (1.9165*(math.sin((356.6+(0.9856*DoY))                \
                    * (math.pi/180)))))*(math.pi/180))))*180)               \
                    / math.pi

    # equation of time #####################################################
    # (More info on http://www.srrb.noaa.gov/highlights/sunrise/azel.html)
    eqt             = (((5.0323-(430.847*math.cos((((2*math.pi)*DoY)/366)+4.8718)))\
                    + (12.5024*(math.cos(2*((((2*math.pi)*DoY)/366)+4.8718))))\
                    + (18.25*(math.cos(3*((((2*math.pi)*DoY)/366)+4.8718))))\
                    - (100.976*(math.sin((((2*math.pi)*DoY)/366)+4.8718))))\
                    + (595.275*(math.sin(2*((((2*math.pi)*DoY)/366)+4.8718))))\
                    + (3.6858*(math.sin(3*((((2*math.pi)*DoY)/366)+4.871))))\
                    - (12.47*(math.sin(4*((((2*math.pi)*DoY)/366)+4.8718)))))\
                    / 60
    ## print "Equation of time                             : %s min" % eqt
    # time of solar noon ###################################################
    sol_noon        = ((12+dst_off)-(eqt/60))-((tz_off_deg-lon)/15)
    ## print "Solar Noon                                   : %s " % sol_noon
    # solar zenith angle in DEG ############################################
    sol_zen         = math.acos(((math.sin(lat*(math.pi/180)))              \
                    * (math.sin(sun_decl*(math.pi/180))))                   \
                    + (((math.cos(lat*((math.pi/180))))                     \
                    * (math.cos(sun_decl*(math.pi/180))))                   \
                    * (math.cos((ToD-sol_noon)*(math.pi/12)))))             \
                    * (180/math.pi)
    # in extreme latitude, values over 90 may occurs.
    #if sol_zen > 90:
    # barometric pressure of the measurement site
    # (this should be replaced by the real measured value) in kPa
    atm_press       = 101.325                                               \
                    * math.pow(((288-(0.0065*(alt-0)))/288)                 \
                    , (9.80665/(0.0065*287)))
    atm_press=100.5
    ## print "Estimated Barometric Pressure at site        : %s kPa" % atm_press
    # Estimated air vapor pressure in kPa ###################################
    atm_vapor_press = (0.61121*math.exp((17.502*atm_temp)                   \
                    / (240.97+atm_temp)))                                   \
                    * (atm_hum/100)
    ## print "Estimated Vapor Pressure at site             : %s kPa" % atm_vapor_press
    # extraterrestrial radiation in W/m2 ###################################
    toa = (sol_const*sun_rel_dist_f)                            \
                    * (math.cos(sol_zen*(math.pi/180)))

    if toa < 0:
        toa = 0

    return toa

def TOA_fast(lons, lats, dates):
    SOL_CONST = 1367.8
    toa = np.zeros((dates.size, lats.size, lons.size))
    for (d, la, lo), v in np.ndenumerate(toa):
        tz_offset = int(lons[lo] / 15)
        hour = dates[d].hour
        #local_hour = (hour + tz_offset) % 24
        doy = dates[d].dayofyear
        lon = lons[lo]
        lat = lats[la]
        sun_rel_dist_f = 1.0 / (1.0 - 9.464e-4 * np.sin(doy)
                         - 0.01671 * np.cos(doy)
                         - 1.489e-4 * np.cos(2.0 * doy) - 2.917e-5 * np.sin(3.0 * doy)
                         - 3.438e-4 * np.cos(4.0 * doy)) ** 2
        sun_decl = (np.arcsin(0.39785 * (np.sin(((278.97 + (0.9856 * doy)) +
                        (1.9165 * (np.sin((356.6 + (0.9856 * doy)) *
                        (np.pi / 180))))) * (np.pi / 180)))) * 180) / np.pi
        eqtime = (((5.0323-(430.847*np.cos((((2*np.pi)*doy)/366)+4.8718)))\
                    + (12.5024*(np.cos(2*((((2*np.pi)*doy)/366)+4.8718))))\
                    + (18.25*(np.cos(3*((((2*np.pi)*doy)/366)+4.8718))))\
                    - (100.976*(np.sin((((2*np.pi)*doy)/366)+4.8718))))\
                    + (595.275*(np.sin(2*((((2*np.pi)*doy)/366)+4.8718))))\
                    + (3.6858*(np.sin(3*((((2*np.pi)*doy)/366)+4.8718))))\
                    - (12.47*(np.sin(4*((((2*np.pi)*doy)/366)+4.8718)))))\
                    / 60
        #sol_noon = (720 + 4 * lon - eqtime) / 60.0
        sol_noon = ((12 - tz_offset)-(eqtime/60))
        sol_zen = np.arccos(((np.sin(lat * (np.pi / 180)))
                    * (np.sin(sun_decl * (np.pi / 180))))
                    + (((np.cos(lat * ((np.pi / 180))))
                    * (np.cos(sun_decl * (np.pi / 180))))
                    * (np.cos((hour - sol_noon) * (np.pi/12))))) \
                    * (180 / np.pi)
        toa[d, la, lo] = SOL_CONST * sun_rel_dist_f * np.cos(sol_zen * (np.pi/180))
    toa[toa < 0] = 0
    return toa

if __name__ == "__main__":
    lon = np.arange(-104,-100,0.1)
    lat = np.arange(35,38,0.1)
    dates = pd.date_range('2014-10-1 00:00','2014-10-03 23:00', freq="H")
    toa = TOA_fast(lon, lat, dates)
    plt.plot(dates.values,toa[:, 10 , 15].flatten())
    plt.show()