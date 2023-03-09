import calibration as calibrate
import background_subtraction as bs


if __name__ == '__main__':
    # calibrate.calibrateExtrinsic()
    # calibrate.saveFrame()
    # calibrate.createLookupTable()
    bs.createBackgroundModel()
    bs.GenerateForeground()
