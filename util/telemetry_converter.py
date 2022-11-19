# created by Steffen Urban, 2021

import json
import numpy as np
from csv import reader

class TelemetryImporter:
    ''' TelemetryImporter

    '''
    def __init__(self):    
        self.ms_to_sec = 1e-3
        self.us_to_sec = 1e-6
        self.ns_to_sec = 1e-9

        self.telemetry = {}

    def _remove_seconds(self, accl, gyro, timestamps_ns, skip_seconds):
        skip_ns = skip_seconds / self.ns_to_sec

        ds = timestamps_ns[1] - timestamps_ns[0]
        nr_remove = round(skip_ns / ds)

        accl = accl[nr_remove:len(timestamps_ns) - nr_remove]
        gyro = gyro[nr_remove:len(timestamps_ns) - nr_remove]

        timestamps_ns = timestamps_ns[nr_remove:len(timestamps_ns) - nr_remove]

        return accl, gyro, timestamps_ns

    def read_gopro_telemetry(self, path_to_jsons, skip_seconds=0.0):
        '''
        path_to_jsons : path to json file or list of paths for multiple files
        skip_seconds : float
            How many seconds to cut from beginning and end of stream
        '''
        
        if isinstance(path_to_jsons, (list, tuple)):
            accl = []
            gyro = []
            timestamps_ns = []
            image_timestamps_ns = []
            last_timestamp, last_img_timestamp = 0.0, 0.0

            for p in path_to_jsons:
                telemetry = self._read_gopro_telemetry(p, skip_seconds=0.0)
                accl.extend(telemetry["accelerometer"])
                gyro.extend(telemetry["gyroscope"])
                times = last_timestamp + np.asarray(telemetry["timestamps_ns"])
                img_times = last_img_timestamp + np.asarray(telemetry["img_timestamps_ns"])

                last_img_timestamp = img_times[-1]
                last_timestamp = times[-1]
                print("Setting last sensor time to: ",last_timestamp)
                print("Setting last image time to: ",last_img_timestamp)

                timestamps_ns.extend(times.tolist())
                image_timestamps_ns.extend(img_times.tolist())
            if skip_seconds != 0.0:
                accl, gyro, timestamps_ns = self._remove_seconds(accl, gyro, timestamps_ns, skip_seconds)
                accl = accl[0:len(timestamps_ns)]
                gyro = gyro[0:len(timestamps_ns)]
            
            self.telemetry["accelerometer"] = accl
            self.telemetry["gyroscope"] = gyro
            self.telemetry["timestamps_ns"] = timestamps_ns
            self.telemetry["img_timestamps_ns"] = image_timestamps_ns
            self.telemetry["camera_fps"] = telemetry["camera_fps"]
        else:
            self.telemetry = self._read_gopro_telemetry(path_to_jsons, skip_seconds=skip_seconds)

    '''
    path_to_json : str 
        path to json file
    skip_seconds : float
        How many seconds to cut from beginning and end of stream
    '''
    def _read_gopro_telemetry(self, path_to_json, skip_seconds=0.0):

        with open(path_to_json, 'r') as f:
            json_data = json.load(f)

        accl, gyro, cori, gravity  = [], [], [], []
        timestamps_ns, cori_timestamps_ns, gps_timestamps_ns = [], [], []
        gps_llh, gps_prec = [], []

        for a in json_data['1']['streams']['ACCL']['samples']:
            timestamps_ns.append(a['cts'] * self.ms_to_sec / self.ns_to_sec)
            accl.append([a['value'][1], a['value'][2], a['value'][0]])
        for g in json_data['1']['streams']['GYRO']['samples']:
            gyro.append([g['value'][1], g['value'][2], g['value'][0]])
        # image orientation at framerate
        for c in json_data['1']['streams']['CORI']['samples']:
            # order w,x,z,y https://github.com/gopro/gpmf-parser/issues/100#issuecomment-656154136
            w, x, z, y = c['value'][0], c['value'][1], c['value'][2], c['value'][3]
            cori.append([x, y, z, w])
            cori_timestamps_ns.append(c['cts'] * self.ms_to_sec / self.ns_to_sec)
        
        # gravity vector in camera coordinates at framerate
        for g in json_data['1']['streams']['GRAV']['samples']:
            gravity.append([g['value'][0], g['value'][1], g['value'][2]])
        
        # GPS
        for g in json_data["1"]["streams"]["GPS5"]["samples"]:
            gps_timestamps_ns.append(g['cts'] * self.ms_to_sec / self.ns_to_sec)
            lat, long, alt = g["value"][0], g["value"][1], g["value"][2]
            gps_llh.append([lat,long,alt])
            gps_prec.append(g["precision"])

        camera_fps = json_data['frames/second']
        if skip_seconds != 0.0:
            accl, gyro, timestamps_ns = self._remove_seconds(accl, gyro, timestamps_ns, skip_seconds)

        accl = accl[0:len(timestamps_ns)]
        gyro = gyro[0:len(timestamps_ns)]

        telemetry = {}
        telemetry["accelerometer"] = accl
        telemetry["gyroscope"] = gyro
        telemetry["timestamps_ns"] = timestamps_ns
        telemetry["camera_fps"] = camera_fps
        telemetry["gravity"] = gravity 
        telemetry["camera_orientation"] = cori
        telemetry["img_timestamps_ns"] = cori_timestamps_ns

        telemetry["gps_llh"] = gps_llh
        telemetry["gps_precision"] = gps_prec
        telemetry["gps_timestamps_ns"] = gps_timestamps_ns
        return telemetry

    def get_gps_pos_at_frametimes(self, img_times_ns=None):
        '''
        Interpolate a GPS coordinate for each frame.
        Probably not very accurate but ...
        '''
        import pymap3d
        # interpolate camera gps info at frametimes
        frame_gps_ecef = [
            pymap3d.geodetic2ecef(llh[0],llh[1],llh[2]) for llh in self.telemetry["gps_llh"]]
        frame_gps_ecef = np.array(frame_gps_ecef)
        gps_times = np.array(self.telemetry["gps_timestamps_ns"]) * self.ns_to_sec
        if img_times_ns is not None:
            frame_times = np.array(img_times_ns) * self.ns_to_sec
        else:
            frame_times = np.array(self.telemetry["image_timestamps_ns"]) * self.ns_to_sec
        
        # find valid interval (interpolate only where we actually have gps measurements)
        start_frame_time_idx = np.where(gps_times[0] < frame_times)[0][0]
        end_frame_time_idx = np.where(gps_times[-1] <= frame_times)[0]
        if not end_frame_time_idx:
            end_frame_time_idx = len(frame_times)

        cam_hz = 1 / self.telemetry["camera_fps"]
        if img_times_ns is not None:
            interp_frame_times = frame_times[start_frame_time_idx:end_frame_time_idx]
        else:
            interp_frame_times = np.round(
            np.arange(
                np.round(frame_times[start_frame_time_idx],2), 
                np.round(frame_times[end_frame_time_idx],2) - cam_hz, cam_hz) ,3).tolist()

        x_interp = np.interp(interp_frame_times, gps_times, frame_gps_ecef[:,0])
        y_interp = np.interp(interp_frame_times, gps_times, frame_gps_ecef[:,1])
        z_interp = np.interp(interp_frame_times, gps_times, frame_gps_ecef[:,2])
        prec_interp = np.interp(interp_frame_times, gps_times, self.telemetry["gps_precision"])
        xyz_interp = np.stack([x_interp,y_interp,z_interp],1)

        camera_gps = dict(zip((np.array(interp_frame_times)*1e9).astype(np.int), xyz_interp.tolist()))

        return camera_gps, prec_interp