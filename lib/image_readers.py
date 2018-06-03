import time
import os

from image_names import image2template_directory, find_matching_images, \
    template_directory_number2image

from run_job import run_job

def check_file_readable(filename):
    '''Check that the file filename exists and that it can be read. Returns
    only if everything is OK.'''

    if not os.path.exists(filename):
        raise RuntimeError, 'file %s not found' % filename

    if not os.access(filename, os.R_OK):
        raise RuntimeError, 'file %s not readable' % filename

    return

def get_dectris_serial_no(record):
    if not 'S/N' in record:
        return '0'
    tokens = record.split()
    return tokens[tokens.index('S/N') + 1]

def failover_h5(h5_master_file):
    """Added by Y.Yamada"""

    header = {}

    header['two_theta'] = 0.0

    import h5py

    h5file = h5py.File(h5_master_file, "r")

    if h5file['/entry/instrument/detector/description'].value == 'Dectris Eiger 4M':
        header['detector_class'] = 'eiger 4M'
        header['detector'] = 'dectris'
        header['size'] = (2167, 2070)

    if h5file['/entry/instrument/detector/description'].value == 'Dectris Eiger 9M':
        header['detector_class'] = 'eiger 9M'
        header['detector'] = 'dectris'
        header['size'] = (3269, 3110)

    if h5file['/entry/instrument/detector/description'].value == 'Dectris Eiger 16M':
        header['detector_class'] = 'eiger 16M'
        header['detector'] = 'dectris'
        header['size'] = (4371, 4150)

    header['phi_start'] = h5file["/entry/sample/goniometer/omega_start"].value
    header['phi_width'] = h5file["/entry/sample/goniometer/omega_increment"].value
    header['phi_end'] = header['phi_start'] + header['phi_width']
    header['oscillation'] = header['phi_start'], header['phi_width']
    header['exposure_time'] = h5file["/entry/instrument/detector/frame_time"].value
    header['distance'] = 1000 * h5file["/entry/instrument/detector/detector_distance"].value
    header['wavelength'] = h5file["/entry/instrument/beam/incident_wavelength"].value

    header['pixel'] = 1000 * h5file["/entry/instrument/detector/x_pixel_size"].value, \
                      1000 * h5file["/entry/instrument/detector/y_pixel_size"].value
    header['saturation'] = h5file["/entry/instrument/detector/detectorSpecific/countrate_correction_count_cutoff"].value
    header['sensor'] = 1000 * h5file["/entry/instrument/detector/sensor_thickness"].value
    beam_pixels = h5file['/entry/instrument/detector/beam_center_x'].value, \
                  h5file['/entry/instrument/detector/beam_center_y'].value
    # for CBF images need to swap these to put in XDS mosflm
    # coordinate frame...
    header['beam'] = beam_pixels[0] * header['pixel'][0], \
                     beam_pixels[1] * header['pixel'][1]

    datestring = h5file['/entry/instrument/detector/detectorSpecific/data_collection_date'].value.split(".")[0]
    format = '%Y-%m-%dT%H:%M:%S'
    struct_time = time.strptime(datestring, format)
    header['date'] = time.asctime(struct_time)
    header['epoch'] = time.mktime(struct_time)

    return header

def failover_cbf(cbf_file):
    '''CBF files from the latest update to the PILATUS detector cause a
    segmentation fault in diffdump. This is a workaround.'''

    header = { }

    header['two_theta'] = 0.0

    for record in open(cbf_file):
        if '_array_data.data' in record:
            break

        if 'EIGER 1M' in record:
            header['detector_class'] = 'eiger 1M'
            header['detector'] = 'dectris'
            header['size'] = (1065, 1030)
            header['serial_number'] = get_dectris_serial_no(record)

        if 'PILATUS 2M' in record:
            header['detector_class'] = 'pilatus 2M'
            header['detector'] = 'dectris'
            header['size'] = (1679, 1475)
            header['serial_number'] = get_dectris_serial_no(record)
            continue

        if 'PILATUS 6M' in record:
            header['detector_class'] = 'pilatus 6M'
            header['detector'] = 'dectris'
            header['size'] = (2527, 2463)
            header['serial_number'] = get_dectris_serial_no(record)
            continue

        if 'PILATUS3 6M' in record:
            header['detector_class'] = 'pilatus 6M'
            header['detector'] = 'dectris'
            header['size'] = (2527, 2463)
            header['serial_number'] = get_dectris_serial_no(record)
            continue

        if 'Start_angle' in record:
            header['phi_start'] = float(record.split()[-2])
            continue

        if 'Angle_increment' in record:
            header['phi_width'] = float(record.split()[-2])
            header['phi_end'] = header['phi_start'] + header['phi_width']
            header['oscillation'] = header['phi_start'], header['phi_width']
            continue

        if 'Exposure_period' in record:
            header['exposure_time'] = float(record.split()[-2])
            continue

        if 'Detector_distance' in record:
            header['distance'] = 1000 * float(record.split()[2])
            continue

        if 'Wavelength' in record:
            header['wavelength'] = float(record.split()[-2])
            continue

        if 'Pixel_size' in record:
            header['pixel'] = 1000 * float(record.split()[2]), \
                              1000 * float(record.split()[5])
            continue

        if 'Count_cutoff' in record:
            header['saturation'] = int(record.split()[2])

        if 'Silicon sensor' in record:
            header['sensor'] = 1000 * float(record.split()[4])

        if 'Beam_xy' in record:
            beam_pixels = map(float, record.replace('(', '').replace(
                ')', '').replace(',', '').split()[2:4])

            # for CBF images need to swap these to put in XDS mosflm
            # coordinate frame...
            header['beam'] = beam_pixels[0] * header['pixel'][0], \
                             beam_pixels[1] * header['pixel'][1]

            continue

        # try to get the date etc. literally.

        try:
            datestring = record.split()[-1].split('.')[0]
            format = '%Y-%b-%dT%H:%M:%S'
            struct_time = time.strptime(datestring, format)
            header['date'] = time.asctime(struct_time)
            header['epoch'] = time.mktime(struct_time)

        except:
            pass

        try:
            datestring = record.replace('#', '').strip().split('.')[0]
            format = '%Y/%b/%d %H:%M:%S'
            struct_time = time.strptime(datestring, format)
            header['date'] = time.asctime(struct_time)
            header['epoch'] = time.mktime(struct_time)

        except:
            pass

    # cope with vertical goniometer on I24 @ DLS from 2015/1/1
    if header.get('serial_number', '0') == '60-0119' and \
            int(header['date'].split()[-1]) >= 2015:
        header['goniometer_is_vertical'] = True
    else:
        header['goniometer_is_vertical'] = False

    return header

def read_image_metadata(image):
    '''Read the image header and send back the resulting metadata in a
    dictionary.'''

    assert(os.path.exists(image))

    # FIXME also check that the file can also be read - assert is acceptable,
    # also use the first image in the wedge to get the frame metadata

    template, directory = image2template_directory(image)
    matching = find_matching_images(template, directory)
    image = template_directory_number2image(template, directory, min(matching))

    # work around (preempt) diffdump failure with the new 2M instrument
    # FIXME may also need to do this for the new 6M instrument which is
    # incoming...

    try:
        if '.cbf' in image[-4:]:
            metadata = failover_cbf(image)

            assert(metadata['detector_class'] in \
                   ['pilatus 2M', 'pilatus 6M', 'eiger 1M'])

            if metadata['detector_class'] == 'pilatus 2M':
                metadata['detector'] = 'PILATUS_2M'
            elif metadata['detector_class'] == 'eiger 1M':
                metadata['detector'] = 'EIGER_1M'
            else:
                metadata['detector'] = 'PILATUS_6M'

            # handle I24 @ DLS vertical goniometer from 2015/1/1
            if metadata['goniometer_is_vertical']:
                metadata['detector'] = '%sV' % metadata['detector']

            metadata['directory'] = directory
            metadata['template'] = template
            metadata['start'] = min(matching)
            metadata['end'] = max(matching)

            metadata['matching'] = matching

            return metadata

        if 'master.h5' in image[-9:]:
            metadata = failover_h5(image)
            assert(metadata['detector_class'] in \
                    ['eiger 4M', 'eiger 9M', 'eiger 16M'])

            if metadata['detector_class'] == 'eiger 4M':
                metadata['detector'] = 'EIGER_4M'
            elif metadata['detector_class'] == 'eiger 9M':
                metadata['detector'] = 'EIGER_9M'
            else:
                metadata['detector'] = 'EIGER_16M'

            metadata['directory'] = directory
            metadata['template'] = template
            metadata['start'] = min(matching)
            metadata['end'] = max(matching)

            metadata['matching'] = matching

            return metadata

    except ValueError, e:
        pass

    # MAR CCD images record the beam centre in pixels...

    diffdump_output = run_job('diffdump', arguments = [image])

    metadata = { }

    for record in diffdump_output:
        if 'Wavelength' in record:
            wavelength = float(record.split()[-2])
            metadata['wavelength'] = wavelength

        elif 'Beam center' in record:
            x = float(record.replace('(', ' ').replace(
                'mm', ' ').replace(',', ' ').split()[3])
            y = float(record.replace('(', ' ').replace(
                'mm', ' ').replace(',', ' ').split()[4])
            metadata['beam'] = x, y

        elif 'Image Size' in record:
            x = int(record.replace('(', ' ').replace(
                'px', ' ').replace(',', ' ').split()[3])
            y = int(record.replace('(', ' ').replace(
                'px', ' ').replace(',', ' ').split()[4])
            metadata['size'] = x, y

        elif 'Pixel Size' in record:
            x = float(record.replace('(', ' ').replace(
                'mm', ' ').replace(',', ' ').split()[3])
            y = float(record.replace('(', ' ').replace(
                'mm', ' ').replace(',', ' ').split()[4])
            metadata['pixel'] = x, y

        elif 'Distance' in record:
            distance = float(record.split()[-2])
            metadata['distance'] = distance

        elif 'Exposure time' in record:
            metadata['exposure_time'] = float(record.split()[-2])

        elif 'Oscillation' in record:
            phi_start = float(record.split()[3])
            phi_end = float(record.split()[5])
            phi_width = phi_end - phi_start

            if phi_width > 360.0:
                phi_width -= 360.0

            metadata['phi_start'] = phi_start
            metadata['phi_width'] = phi_width
            metadata['phi_end'] = phi_end
            metadata['oscillation'] = phi_start, phi_width

        elif 'Manufacturer' in record or 'Image type' in record:
            detector = record.split()[-1]
            if detector == 'ADSC':
                metadata['detector'] = 'ADSC'
            elif detector == 'MAR':
                metadata['detector'] = 'MARCCD'
            elif detector == 'DECTRIS':
                metadata['detector'] = 'PILATUS_6M'
            elif detector == 'RIGAKU':
                metadata['detector'] = 'RIGAKU'
            else:
                raise RuntimeError, 'detector %s not yet supported' % \
                      detector

    if (metadata['detector'] == 'PILATUS_6M') and \
       (metadata['size'] == (1679, 1475)):
        metadata['detector'] = 'PILATUS_2M'

    # now compute the filename template and what have you, and also
    # verify that the results stored make sense, particularly w.r.t.
    # the beam centre, which may be stored in pixels not mm.

    template, directory = image2template_directory(image)

    # MAR CCD images record the beam centre in pixels...

    if metadata['detector'] == 'MARCCD':
        metadata['beam'] = (metadata['beam'][0] * metadata['pixel'][0],
                            metadata['beam'][1] * metadata['pixel'][1])

    metadata['directory'] = directory
    metadata['template'] = template
    metadata['start'] = min(matching)
    metadata['end'] = max(matching)
    metadata['matching'] = matching

    return metadata

# FIXME add some unit tests in here...

if __name__ == '__main__':
    import sys
    md = read_image_metadata(sys.argv[1])
    for name in sorted(md):
        print name, md[name]
