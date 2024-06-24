# Weiguang function at https://github.com/weiguangcui/pymsz/blob/master/pymsz/rotate_data.py
import numpy as np
from struct import unpack
from os import fstat
#import matplotlib.pyplot as plt
nmets = 11

"""
def plot_cluster(image,cmap='seismic'):
    fig = plt.figure(figsize=(10, 10))
    im = plt.imshow(image,cmap=cmap)
    fig.colorbar(im)
    plt.show()
"""
def rotate_data(pos, axis, vel=None, bvel=None):
    r""" rotate the data points (3d) to a given direction. Returns data points (2d) at line of sight.
    Parameter:
    ----------
    pos     : input data points in 3D.
    axis    : can be 'x', 'y', 'z' (must be 2D), or a list or numpy array of degrees
              [alpha, beta, gamma], which will rotate the data points by $\alpha$ around
              the x-axis, $\beta$ around the y-axis, and $\gamma$ around the z-axis.
              or a numpy array with the rotation matrix directly, must be 3x3 matrix.
    vel     : 3D velocity of the input data points. Default: None, will return an empty list.
                Otherwise, rotate_data will also return the velocity in the axis direction.
    bvel    : bulk velocity of the cluster in 3D, defualt None, resturn 0 for the bulk velocity
              along line of sight. If it is not None, bulk velocity along line of sight will be return.
    Notes:
    --------
    When you have vel is not None, the function will return two arrays: pos, vel in axis direction.
    This function does not work with yt data currrently.
    """

    nvel = []; nbvel = 0;
    if isinstance(axis, type('')):
        npos = np.copy(pos)
        if axis.lower() == 'y':  # x-z plane
            npos[:, 1] = pos[:, 2]
            npos[:, 2] = pos[:, 1]
            if vel is not None:
                nvel = vel[:, 1]
            if bvel is not None:
                nbvel = bvel[1]
        elif axis.lower() == 'x':  # y - z plane
            npos[:, 0] = pos[:, 1]
            npos[:, 1] = pos[:, 2]
            npos[:, 2] = pos[:, 0]
            if vel is not None:
                nvel = vel[:, 0]
            if bvel is not None:
                nbvel = bvel[0]
        elif axis.lower() == 'z':
            if vel is not None:
                nvel = vel[:, 2]
            if bvel is not None:
                nbvel = bvel[2]
        else:
            raise ValueError("Do not accept this value %s for projection" % axis)
    elif isinstance(axis, type([])):
        if len(axis) == 3:
            sa, ca = np.sin(axis[0] / 180. *
                            np.pi), np.cos(axis[0] / 180. * np.pi)
            sb, cb = np.sin(axis[1] / 180. *
                            np.pi), np.cos(axis[1] / 180. * np.pi)
            sg, cg = np.sin(axis[2] / 180. *
                            np.pi), np.cos(axis[2] / 180. * np.pi)
            # ratation matrix from
            # http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/
            Rxyz = np.array(
                [[cb * cg, cg * sa * sb - ca * sg, ca * cg * sb + sa * sg],
                 [cb * sg, ca * cg + sa * sb * sg, ca * sb * sg - cg * sa],
                 [-sb,     cb * sa,                ca * cb]], dtype=np.float64)
            npos = np.dot(pos, Rxyz)
            if vel is not None:
                nvel = np.dot(vel, Rxyz)#[:, 2]
            if bvel is not None:
                nbvel = np.dot(bvel, Rxyz)[2]
        else:
            raise ValueError("Do not accept this value %s for projection" % axis)
    elif isinstance(axis, type(np.array([]))):
        if len(axis.shape) == 1:
            sa, ca = np.sin(axis[0] / 180. *
                            np.pi), np.cos(axis[0] / 180. * np.pi)
            sb, cb = np.sin(axis[1] / 180. *
                            np.pi), np.cos(axis[1] / 180. * np.pi)
            sg, cg = np.sin(axis[2] / 180. *
                            np.pi), np.cos(axis[2] / 180. * np.pi)
            # ratation matrix from
            # http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/
            Rxyz = np.array(
                [[cb * cg, cg * sa * sb - ca * sg, ca * cg * sb + sa * sg],
                 [cb * sg, ca * cg + sa * sb * sg, ca * sb * sg - cg * sa],
                 [-sb,     cb * sa,                ca * cb]], dtype=np.float64)
            npos = np.dot(pos, Rxyz)
            if vel is not None:
                nvel = np.dot(vel, Rxyz)#[:, 2]
            if bvel is not None:
                nbvel = np.dot(bvel, Rxyz)[2]
        elif len(axis.shape) == 2:
            if axis.shape[0] == axis.shape[1] == 3:
                npos = np.dot(pos, axis)
                if vel is not None:
                    nvel = np.dot(vel, axis)#[:, 2]
                if bvel is not None:
                    nbvel = np.dot(bvel, axis)[2]
            else:
                raise ValueError("Axis shape is not 3x3: ", axis.shape)
        else:
            raise ValueError("Do not accept this shape of axis %s for projection!" % axis)
    else:
        raise ValueError("Do not accept this value %s for projection!" % axis)
    return npos, nvel, nbvel



def readsnapsgl(filename, block, endian=None, quiet=False, longid=False, nmet=11,
                fullmass=False, mu=None, fmt=None, ptype=None, rawdata=False):
    """
    readsnapsgl(filename,block,endian=None,quiet=None,longid=None,met=None, fmt=None)
        read snapshot files and new subfind files, return any block result you need.

    Parameters:
    ---------------
        filename: path plus full file name. e.g.  /your/dir/snap_009.0
        block: The block you want to read, e.g. "HEAD". Look for more info with block == "INFO"
        little endian: ">", big endian : "<", other/default : "=" or "@"
        longid: Is the particle ID saved in long long (uint64)? Default : False
        nmet: Specify how many different matels are produced in the simulation, default: 11
        fullmass: return all mass of particles inorder of saved particle position
                  False(default): return only mass block
        mu: mean_molecular_weight. Specify this value for gas temperature.
                  It will be ignored when you have NE block in your simulatin data.
        fmt: default or 1: G3 format with blocks; 0: G2 format; -1: new subfind results.
        ptype: read only specified particle type: 0: gas, 1: DM, 2: , 3: , 4: star, 5: bh
        rawdata: default False. If True, retrun the binary data in str, which need unpack yourself.

    Notes:
    ------------
    The old parameter met "z", is deprecated. If you need metal in z instead of elements,
    simply put 'Z   ' for the block.

    For these snapshots which are more than 4 Gb, i.e. the data size (bytes) indicator,
    which is just ahead of the data block, is negative, you can use `ptype=1` to overcome
    the error in reading the data.
    """

    if endian is None:
        npf = open(filename, 'rb')
        (rhead,) = unpack('<I', npf.read(4))
        npf.close()
        if (rhead == 134217728) or (rhead == 65536):
            endian = '>'
        elif (rhead == 8) or (rhead == 256):
            endian = '<'
        else:
            raise ValueError("Don't know the file endian with this value %d." % rhead)
        if not quiet:
            print("detected file format endian = ", endian)

    if fmt is None:
        # try to get the format
        npf = open(filename, 'rb')
        bs1 = unpack(endian + 'i', npf.read(4))[0]
        if bs1 == 256:
            fmt = 0
            if not quiet:
                print("Snapshot with Gadget 2 format")
        elif bs1 == 8:
            fmt = 1
            if not quiet:
                print("Snapshot with Gadget 3 format with blocks")
        else:
            print("Not knowing what is this value ", bs1, "still assuming format with block")
            print("This may have incorrect results, better have a check on endian.")
            fmt = 1
        npf.close()

    global nmets
    if nmets != nmet:
        nmets = nmet

    # read header
    npf = open(filename, 'rb')
    if fmt != 0:
        bname, bsize = read_bhead(npf)

    class rhead:
        def __init__(self, npf):
            bs1 = npf.read(4)  # size of header
            del(bs1)
            self.npart = np.zeros(6, dtype='int32')
            self.npart[:] = unpack(endian + 'i i i i i i', npf.read(4 * 6))
            self.masstbl = np.zeros(6, dtype='float64')
            self.masstbl[:] = unpack(endian + 'd d d d d d', npf.read(8 * 6))
            self.Time, self.Redshift = unpack(endian + 'd d', npf.read(2 * 8))
            self.F_sfr, self.F_fb = unpack(endian + 'i i', npf.read(2 * 4))
            self.totnum = np.zeros(6, dtype='int64')
            self.totnum[:] = unpack(endian + 'i i i i i i', npf.read(6 * 4))
            self.F_cool, self.Numfiles = unpack(endian + 'i i', npf.read(2 * 4))
            self.Boxsize, self.Omega0, self.OmegaLambda, self.Hubbleparam = unpack(endian + 'd d d d', npf.read(4 * 8))
            self.F_agn, self.F_metal = unpack(endian + 'i i', npf.read(2 * 4))
            self.nallHW = np.zeros(6, dtype='int32')
            self.nallHW[:] = unpack(endian + 'i i i i i i', npf.read(6 * 4))
            self.F_entr_ics = unpack(endian + 'i', npf.read(4))[0]

    hd = rhead(npf)
    npf.close()

    if block == 'HEAD':
        return hd

    if block == 'IDTP':  # Particle type
        idtype = np.zeros(hd.npart.sum(), dtype=np.int32)
        nn = 0
        for i, j in enumerate(hd.npart):
            if j > 0:
                idtype[nn:nn + j] = i
                nn += j
        return(idtype)
    else:
        if fmt >= 0:
            if ptype is not None:
                if ptype == 0:
                    pty = [0, hd.npart[0]]
                elif ptype == 1:
                    pty = [hd.npart[0], hd.npart[1]]
                elif ptype == 2:
                    pty = [np.sum(hd.npart[:2]), hd.npart[2]]
                elif ptype == 3:
                    pty = [np.sum(hd.npart[:3]), hd.npart[3]]
                elif ptype == 4:
                    pty = [np.sum(hd.npart[:4]), hd.npart[4]]
                elif ptype == 5:
                    pty = [np.sum(hd.npart[:5]), hd.npart[5]]
                else:
                    raise ValueError("Don't accept ptype value %d" % ptype)
            else:
                pty = None  # the same as ptype

            if block == "MASS":
                idg0 = (hd.npart > 0) & (hd.masstbl <= 0)
                if fullmass:
                    if len(hd.npart[idg0]) == 0:  # No Mass block!
                        idg1 = (hd.npart > 0) & (hd.masstbl > 0)
                        if len(hd.npart[idg1]) == 1:
                            return hd.masstbl[idg1]
                        else:  # multi masstble
                            totmass = np.zeros(np.sum(hd.npart, dtype='int64'), dtype='float32')
                            countnm = 0
                            for i in np.arange(6):
                                if hd.npart[i] > 0:
                                    totmass[countnm:countnm + hd.npart[i]] = hd.masstbl[i]
                                    countnm += hd.npart[i]
                            return totmass
                elif ptype is not None:
                    if (hd.npart[ptype] > 0) & (hd.masstbl[ptype] > 0):
                        return hd.masstbl[ptype]
                else:
                    if len(hd.npart[idg0]) == 0:  # No Mass block!
                        return hd.masstbl

        npf = open(filename, 'rb')
        subdata = read_block(npf, block, endian, quiet, longid, fmt, pty, rawdata)
        if subdata is not None:  # we have subdata
            if block == "MASS":  # We fill the mass with the mass tbl value if needed
                npf.close()
                idg0 = (hd.npart > 0) & (hd.masstbl > 0)
                if (len(hd.npart[idg0]) > 0) and (fullmass):
                    totmass = np.zeros(np.sum(hd.npart, dtype='int64'), dtype='float32')
                    bgc = 0
                    subc = 0
                    for k in np.arange(6):
                        if hd.npart[k] > 0:
                            if(hd.masstbl[k] > 0):
                                totmass[bgc:bgc + hd.npart[k]
                                        ] = np.zeros(hd.npart[k], dtype='float32') + hd.masstbl[k]
                            else:
                                totmass[bgc:bgc + hd.npart[k]] = subdata[subc:subc + hd.npart[k]]
                                subc += hd.npart[k]
                            bgc += hd.npart[k]
                    return totmass
                else:
                    if ptype is not None:
                        if (hd.npart[ptype] == 0) or (hd.masstbl[ptype] > 0):
                            print("This is can not be! hd.npart[ptype] is ",
                                  hd.npart[ptype], "masstbl[ptype] is ", hd.masstbl[ptype])
                            print("I return 0")
                            return(0)
                        else:
                            startc = 0
                            endc = 0
                            for ii in range(ptype + 1):
                                if (hd.npart[ii] > 0) and (hd.masstbl[ii] <= 0):
                                    startc = endc
                                    endc += hd.npart[ii]
                            return(subdata[startc:endc])
                    return subdata
            elif ((block == "Z   ") or (block == "ZTOT") or (block == "Zs  ")) and (ptype is not None):
                if ptype == 0:
                    return subdata[:hd.npart[0]]
                elif ptype == 4:
                    return subdata[hd.npart[0]:]
                else:
                    raise ValueError(
                        "The given ptype %d is not accepted for metallicity block %s.", ptype, block)
            else:
                npf.close()
                return subdata
        else:  # No subdata returned
            if block == 'TEMP':  # No temperature block. Try to calculate the temperature from U
                temp = read_block(npf, "U   ", endian, 1, longid, fmt, pty, rawdata)
                if temp is None:
                    print("Can't read gas Temperature (\"TEMP\") and internal energy (\"U   \")!!")
                else:
                    xH = 0.76  # hydrogen mass-fraction
                    yhelium = (1. - xH) / (4 * xH)
                    NE = read_block(npf, "NE  ", endian, 1, longid, fmt, pty, rawdata)
                    if NE is None:
                        # we assume it is NR run with full ionized gas n_e/nH = 1 + 2*nHe/nH
                        if mu is None:
                            mean_mol_weight = (1. + 4. * yhelium) / (1. + 3 * yhelium + 1)
                        else:
                            mean_mol_weight = mu
                    else:
                        mean_mol_weight = (1. + 4. * yhelium) / (1. + yhelium + NE)
                    v_unit = 1.0e5 * np.sqrt(hd.Time)       # (e.g. 1.0 km/sec)
                    prtn = 1.67373522381e-24  # (proton mass in g)
                    bk = 1.3806488e-16        # (Boltzman constant in CGS)
                    npf.close()
                    return(temp * (5. / 3 - 1) * v_unit**2 * prtn * mean_mol_weight / bk)
            elif ((block == "Z   ") or (block == "ZTOT")):
                # no "Z   " in the data, which needs to calculate it from "Zs  " block
                subdata = read_block(npf, "Zs  ", endian, True, longid, fmt, pty, rawdata)
                if subdata is None:
                    raise ValueError("Can't find the 'Zs  ' block for calculate metallicity!")
                if ptype == 0:
                    if hd.masstbl[0] > 0:
                        mass = np.zeros(hd.npart[0], dtype=hd.masstbl.dtype) + hd.masstbl[0]
                    else:
                        mass = read_block(npf, "MASS", endian, True, longid,
                                          fmt, [0, 0], rawdata)[0:hd.npart[0]]
                    npf.close()
                    # return
                    # np.sum(subdata[0:hd.npart[0],1:],axis=1)/(mass[0:hd.npart[0]]-np.sum(subdata[0:hd.npart[0],:],axis=1))
                    # old version with z = M_z/M_H why?
                    # MASS block do not accept pty, all mass are returned!
                    return np.sum(subdata[0:hd.npart[0], 1:], axis=1) / mass
                elif ptype == 4:
                    # have to use initial mass because the metal block include SN metals.
                    im = read_block(npf, "iM  ", endian, True, longid, fmt, pty, rawdata)
                    npf.close()
                    # return
                    # np.sum(subdata[hd.npart[0]:,1:],axis=1)/(im-np.sum(subdata[hd.npart[0]:,:],axis=1))
                    # old version with z = M_z/M_H why?
                    return np.sum(subdata[hd.npart[0]:, 1:], axis=1) / im
                else:
                    zs = np.zeros(hd.npart[0] + hd.npart[4], dtype=subdata.dtype)
                    if hd.masstbl[0] > 0:
                        mass = np.zeros(hd.npart[0], dtype=hd.masstbl.dtype) + hd.masstbl[0]
                    else:
                        mass = read_block(npf, "MASS", endian, True, longid,
                                          fmt, [0, 0], rawdata)[0:hd.npart[0]]
                    # zs[0:hd.npart[0]]=np.sum(subdata[0:hd.npart[0],1:],axis=1)/(mass[0:hd.npart[0]]-np.sum(subdata[0:hd.npart[0],:],axis=1))
                    # old version
                    zs[0:hd.npart[0]] = np.sum(subdata[0:hd.npart[0], 1:], axis=1) / mass

                    im = read_block(npf, "iM  ", endian, True, longid, fmt, pty, rawdata)
                    # zs[hd.npart[0]:]=np.sum(subdata[hd.npart[0]:,1:],axis=1)/(im-np.sum(subdata[hd.npart[0]:,:],axis=1))
                    zs[hd.npart[0]:] = np.sum(subdata[hd.npart[0]:, 1:], axis=1) / im
                    mass, im, subdata = 0, 0, 0
                    npf.close()
                    return zs

            if not quiet:
                print("No such blocks!!! or Not add in this reading!!!", block)
            npf.close()
            return(0)


# Read Block
def read_block(npf, block, endian, quiet, longid, fmt, pty, rawdata):
    global nmets
    endf = fstat(npf.fileno()).st_size

    bname = 'BLOCK_NAME'
    if fmt == 0:
        npf.seek(8 + 256)  # skip block(16) + header (264)
    elif fmt == 1:
        npf.seek(16 + 8 + 256)  # skip header (264)
    loopnum = 0
    # while bname!='EOFL' :   #Ending block
    while npf.tell() < endf:  # End of file
        if fmt != 0:
            bname, bsize = read_bhead(npf)
            bsize = npf.read(4)
            bsize = unpack(endian + 'i', bsize)[0]
            npf.seek(npf.tell() - 4)
        else:
            bsize = npf.read(4)
            bsize = unpack(endian + 'i', bsize)[0]
            npf.seek(npf.tell() - 4)

            if (block == 'POS ') and (loopnum == 0):
                return read_bdata(npf, 3, np.dtype('float32'), endian, pty)
            elif (block == 'VEL ') and (loopnum == 1):
                return read_bdata(npf, 3, np.dtype('float32'), endian, pty)
            elif (block == 'ID  ') and (loopnum == 2):
                if longid:
                    return read_bdata(npf, 1, np.dtype('uint64'), endian, pty)
                else:
                    return read_bdata(npf, 1, np.dtype('uint32'), endian, pty)
            elif (block == 'MASS') and (loopnum == 3):
                return read_bdata(npf, 1, np.dtype('float32'), endian)
            elif (block == 'U   ') and (loopnum == 4):
                return read_bdata(npf, 1, np.dtype('float32'), endian)
            elif loopnum > 4:
                return None
            loopnum += 1

        if not quiet:
            if fmt != 0:
                print(bname, bsize)
            else:
                print("Format 0, reading block ", block,  "skiping", bsize)

        # For reading snapshot files###
        if rawdata:
            if bname == block:
                return npf.read(unpack(endian + 'i', npf.read(4))[0])

        if bname == block == 'POS ':
            return read_bdata(npf, 3, np.dtype('float32'), endian, pty)
        elif bname == block == 'VEL ':
            return read_bdata(npf, 3, np.dtype('float32'), endian, pty)
        elif bname == block == 'ID  ':
            if longid:
                return read_bdata(npf, 1, np.dtype('uint64'), endian, pty)
            else:
                return read_bdata(npf, 1, np.dtype('uint32'), endian, pty)
        elif bname == block == 'MASS':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'RHO ':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'NE  ':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'NH  ':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'SFR ':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'AGE ':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'POT ':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'iM  ':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'Zs  ':
            return read_bdata(npf, nmets, np.dtype('float32'), endian)
        elif bname == block == 'ZS  ':
            return read_bdata(npf, nmets, np.dtype('float32'), endian)
        elif bname == block == 'HOTT':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'Z   ':  # specified block, which saves the metallicity z
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'ZTOT':  # specified block, which saves the metallicity z
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'CLDX':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'MHI ':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'TEMP':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'HSML':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'PTYP':
            return read_bdata(npf, 1, np.dtype('int32'), endian)
        # Internal energy###
        elif bname == block == 'U   ':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        # INFO ####
        elif bname == block == 'INFO':  # This is print out not return in array
            bs1 = unpack(endian + 'i', npf.read(4))[0]
            buf = npf.read(bs1)
            print("Block   DataType   dim  Type0 Type1 Type2 Type3 Type4 Type5")
            cc = 0
            while cc < bs1:
                print(unpack(endian + '4s 8s i i i i i i i', buf[cc:cc + 40]))
                cc += 40
            return(1)

        # For reading new subfind files###
        elif bname == block == 'GLEN':
            return read_bdata(npf, 1, np.dtype('int32'), endian)
        elif bname == block == 'GOFF':
            return read_bdata(npf, 1, np.dtype('int32'), endian)
        elif bname == block == 'MTOT':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'GPOS':
            return read_bdata(npf, 3, np.dtype('float32'), endian)
        elif bname == block == 'MVIR':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'RVIR':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'M25K':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'R25K':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'M500':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'R500':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'MGAS':
            return read_bdata(npf, 3, np.dtype('float32'), endian)
        elif bname == block == 'MSTR':
            return read_bdata(npf, 3, np.dtype('float32'), endian)
        elif bname == block == 'TGAS':
            return read_bdata(npf, 3, np.dtype('float32'), endian)
        elif bname == block == 'LGAS':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'NCON':
            return read_bdata(npf, 1, np.dtype('int32'), endian)
        elif bname == block == 'MCON':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'BGPO':
            return read_bdata(npf, 3, np.dtype('float32'), endian)
        elif bname == block == 'BGMA':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'BGRA':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'NSUB':
            return read_bdata(npf, 1, np.dtype('int32'), endian)
        elif bname == block == 'FSUB':
            return read_bdata(npf, 1, np.dtype('int32'), endian)
        elif bname == block == 'SLEN':
            return read_bdata(npf, 1, np.dtype('int32'), endian)
        elif bname == block == 'SOFF':
            return read_bdata(npf, 1, np.dtype('int32'), endian)
        elif bname == block == 'SSUB':
            return read_bdata(npf, 1, np.dtype('int32'), endian)
        elif bname == block == 'MSUB':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'SPOS':
            return read_bdata(npf, 3, np.dtype('float32'), endian)
        elif bname == block == 'SVEL':
            return read_bdata(npf, 3, np.dtype('float32'), endian)
        elif bname == block == 'SCM ':
            return read_bdata(npf, 3, np.dtype('float32'), endian)
        elif bname == block == 'SPIN':
            return read_bdata(npf, 3, np.dtype('float32'), endian)
        elif bname == block == 'DSUB':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'VMAX':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'RMAX':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'MBID':
            if longid:
                return read_bdata(npf, 1, np.dtype('int64'), endian)
            else:
                return read_bdata(npf, 1, np.dtype('uint32'), endian)
        elif bname == block == 'GRNR':
            return read_bdata(npf, 1, np.dtype('int32'), endian)
        elif bname == block == 'SMST':
            return read_bdata(npf, 6, np.dtype('float32'), endian)
        elif bname == block == 'SLUM':
            return read_bdata(npf, 12, np.dtype('float32'), endian)
        elif bname == block == 'SLAT':
            return read_bdata(npf, 12, np.dtype('float32'), endian)
        elif bname == block == 'SLOB':
            return read_bdata(npf, 12, np.dtype('float32'), endian)
        elif bname == block == 'DUST':
            return read_bdata(npf, 11, np.dtype('float32'), endian)
        elif bname == block == 'SAGE':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'SZ  ':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'SSFR':
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif bname == block == 'PID ':
            if longid:
                return read_bdata(npf, 1, np.dtype('int64'), endian)
            else:
                return read_bdata(npf, 1, np.dtype('uint32'), endian)

        # For reading my PIAO outputs ###
        elif (bname == block == 'GOFF') or (bname == block == 'GHED') or \
            (bname == block == 'GSBL') or (bname == block == 'GSBO') or \
                (bname == block == 'SBLN') or (bname == block == 'SBOF'):
            return read_bdata(npf, 1, np.dtype('int32'), endian)
        elif (bname == block == 'GMAS') or (bname == block == 'GRAD') or (bname == block == 'SBMS'):
            return read_bdata(npf, 1, np.dtype('float32'), endian)
        elif (bname == block == 'GDCP') or (bname == block == 'GMCP') or (bname == block == 'SBPS'):
            return read_bdata(npf, 3, np.dtype('float32'), endian)
        elif (bname == block == 'GCID') or (bname == block == 'GIDS') or (bname == block == 'SBCI'):
            if longid:
                return read_bdata(npf, 1, np.dtype('int64'), endian)
            else:
                return read_bdata(npf, 1, np.dtype('uint32'), endian)

        else:
            # if fmt != 0:
            #    bsize=unpack(endian+'i',npf.read(4))[0]
            #    npf.seek(bsize+npf.tell()+4)
            # else:
            npf.seek(bsize + 8 + npf.tell())

    return None


# Read Block Head
def read_bhead(npf):
    dummy = npf.read(4)  # dummy
    if dummy == '':
        bname = "EOFL"
        bsize = '0'
    else:
        bname = npf.read(4).decode('ascii')  # label
        bsize = npf.read(4)  # size
        if npf.read(4) != dummy:
            print("header part not consistent!!")
    return bname, bsize


# Read Block data
def read_bdata(npf, column, dt, endian, pty=None):
    bs1 = unpack(endian + 'i', npf.read(4))[0]

    if pty is None:
        buf = npf.read(bs1)
    else:
        npf.seek(npf.tell() + pty[0] * dt.itemsize * column)
        buf = npf.read(pty[1] * dt.itemsize * column)
        bs1 = pty[1] * dt.itemsize * column

    if column == 1:
        arr = np.ndarray(shape=np.int32(bs1 / dt.itemsize), dtype=dt, buffer=buf)
    else:
        arr = np.ndarray(shape=(np.int32(bs1 / dt.itemsize / column), column),
                         dtype=dt, buffer=buf)

    if (endian == '=') or (endian == '<'):  # = and < gives the same result
        return arr
    else:
        return arr.byteswap()


#------------
# Fits
#------------

from astropy.io import fits
from astropy.time import Time

#------------
# Write Fits
#------------


def write_fits_image(img,fname, comments="None", overwrite=False):
    r"""
    Generate a image by binning X-ray counts and write it to a FITS file.
    Parameters
    ----------
    imagefile : string
        The name of the image file to write.
    comments : The comments in str will be put into the fit file header. Defualt: 'None'
                It accepts str or list of str or tuple of str
    overwrite : boolean, optional
        Set to True to overwrite a previous file.
    """
    # import pyfits as pf
    import astropy.io.fits as pf

    if fname[-5:] != ".fits":
        fname = fname + ".fits"


    hdu = pf.PrimaryHDU(img)
    hdu.header["SIMPLE"] = 'T'
    hdu.header.comments["SIMPLE"] = 'conforms to FITS standard'
    hdu.header["BITPIX"] = int(-32)
    hdu.header.comments["BITPIX"] = '32 bit floating point'
    hdu.header["NAXIS"] = int(2)
   # hdu.header["NAXIS1"] = int(self.ydata.shape[0])
   # hdu.header["NAXIS2"] = int(self.ydata.shape[1])
    hdu.header["EXTEND"] = True
    hdu.header.comments["EXTEND"] = 'Extensions may be present'
    hdu.header["RADECSYS"] = 'ICRS    '
    hdu.header.comments["RADECSYS"] = "International Celestial Ref. System"
    hdu.header["CTYPE1"] = 'RA---TAN'
    hdu.header.comments["CTYPE1"] = "Coordinate type"
    hdu.header["CTYPE2"] = 'DEC--TAN'
    hdu.header.comments["CTYPE2"] = "Coordinate type"
    hdu.header["CUNIT1"] = 'deg     '
    hdu.header.comments["CUNIT1"] = 'Units'
    hdu.header["CUNIT2"] = 'deg     '
    hdu.header.comments["CUNIT2"] = 'Units'
   # hdu.header["CRPIX1"] = float(self.npl/2.0)
   # hdu.header.comments["CRPIX1"] = 'X of reference pixel'
    #hdu.header["CRPIX2"] = float(self.npl/2.0)
   # hdu.header.comments["CRPIX2"] = 'Y of reference pixel'
   # hdu.header["CRVAL1"] = float(self.sp[0])
   # hdu.header.comments["CRVAL1"] = 'RA of reference pixel (deg)'
   # hdu.header["CRVAL2"] = float(self.sp[1])
   # hdu.header.comments["CRVAL2"] = 'Dec of reference pixel (deg)'
   # hdu.header["CD1_1"] = -float(self.ar/3600.)
   # hdu.header.comments["CD1_1"] = 'RA deg per column pixel'
    hdu.header["CD1_2"] = float(0)
    hdu.header.comments["CD1_2"] = 'RA deg per row pixel'
    hdu.header["CD2_1"] = float(0)
    hdu.header.comments["CD2_1"] = 'Dec deg per column pixel'
   # hdu.header["CD2_2"] = float(self.ar/3600.)
   # hdu.header.comments["CD2_2"] = 'Dec deg per row pixel'

   # hdu.header["RCVAL1"] = float(self.cc[0])
   # hdu.header.comments["RCVAL1"] = 'Real center X of the data'
   # hdu.header["RCVAL2"] = float(self.cc[1])
   # hdu.header.comments["RCVAL2"] = 'Real center Y of the data'
   # hdu.header["RCVAL3"] = float(self.cc[2])
   # hdu.header.comments["RCVAL3"] = 'Real center Z of the data'
    hdu.header["UNITS"] = "kpc"
    hdu.header.comments["UNITS"] = 'Units for the RCVAL and PSIZE'
    hdu.header["PIXVAL"] = "y parameter"
    hdu.header.comments["PIXVAL"] = 'The y parameter for thermal SZ effect.'
   # hdu.header["ORAD"] = float(self.rr)
   # hdu.header.comments["ORAD"] = 'Rcut in physical for the image.'
   # hdu.header["REDSHIFT"] = float(self.red)
   # hdu.header.comments["REDSHIFT"] = 'The redshift of the object being put to'
   # hdu.header["PSIZE"] = float(self.pxs)
   # hdu.header.comments["PSIZE"] = 'The pixel size in physical at simulation time'

   # hdu.header["AGLRES"] = float(self.ar)
   # hdu.header.comments["AGLRES"] = '\'observation\' angular resolution in arcsec'

    hdu.header["ORIGIN"] = 'Software: DMML'
    hdu.header.comments["ORIGIN"] = 'to be continued'
    #hdu.header["VERSION"] = version.version  # get_property('__version__')
    #hdu.header.comments["VERSION"] = 'Version of the software'
    hdu.header["DATE-OBS"] = Time.now().tt.isot
    if isinstance(comments, type([])) or isinstance(comments, type(())):
        for j in range(len(comments)):
            hdu.header["COMMENT"+str(j+1)] = comments[j]
    elif isinstance(comments, type("")) or isinstance(comments, type('')):
        hdu.header["COMMENT"] = comments
    else:
        raise ValueError("Do not accept this comments type! Please use str or list")
    hdu.writeto(fname, overwrite=overwrite)
    # finish this funtion


def write_fits_image_totalmass(img,fname, comments="None", overwrite=False):
    r"""
    Generate a image by binning X-ray counts and write it to a FITS file.
    Parameters
    ----------
    imagefile : string
        The name of the image file to write.
    comments : The comments in str will be put into the fit file header. Defualt: 'None'
                It accepts str or list of str or tuple of str
    overwrite : boolean, optional
        Set to True to overwrite a previous file.
    """
    # import pyfits as pf
    import astropy.io.fits as pf

    if fname[-5:] != ".fits":
        fname = fname + ".fits"


    hdu = pf.PrimaryHDU(img)
    hdu.header["SIMPLE"] = 'T'
    hdu.header.comments["SIMPLE"] = 'conforms to FITS standard'
    hdu.header["BITPIX"] = int(-32)
    hdu.header.comments["BITPIX"] = '32 bit floating point'
    hdu.header["NAXIS"] = int(2)
   # hdu.header["NAXIS1"] = int(self.ydata.shape[0])
   # hdu.header["NAXIS2"] = int(self.ydata.shape[1])
    hdu.header["EXTEND"] = True
    hdu.header.comments["EXTEND"] = 'Extensions may be present'
    hdu.header["RADECSYS"] = 'ICRS    '
    hdu.header.comments["RADECSYS"] = "International Celestial Ref. System"
    hdu.header["CTYPE1"] = 'RA---TAN'
    hdu.header.comments["CTYPE1"] = "Coordinate type"
    hdu.header["CTYPE2"] = 'DEC--TAN'
    hdu.header.comments["CTYPE2"] = "Coordinate type"
    hdu.header["CUNIT1"] = 'deg     '
    hdu.header.comments["CUNIT1"] = 'Units'
    hdu.header["CUNIT2"] = 'deg     '
    hdu.header.comments["CUNIT2"] = 'Units'
   # hdu.header["CRPIX1"] = float(self.npl/2.0)
   # hdu.header.comments["CRPIX1"] = 'X of reference pixel'
    #hdu.header["CRPIX2"] = float(self.npl/2.0)
   # hdu.header.comments["CRPIX2"] = 'Y of reference pixel'
   # hdu.header["CRVAL1"] = float(self.sp[0])
   # hdu.header.comments["CRVAL1"] = 'RA of reference pixel (deg)'
   # hdu.header["CRVAL2"] = float(self.sp[1])
   # hdu.header.comments["CRVAL2"] = 'Dec of reference pixel (deg)'
   # hdu.header["CD1_1"] = -float(self.ar/3600.)
   # hdu.header.comments["CD1_1"] = 'RA deg per column pixel'
    hdu.header["CD1_2"] = float(0)
    hdu.header.comments["CD1_2"] = 'RA deg per row pixel'
    hdu.header["CD2_1"] = float(0)
    hdu.header.comments["CD2_1"] = 'Dec deg per column pixel'
   # hdu.header["CD2_2"] = float(self.ar/3600.)
   # hdu.header.comments["CD2_2"] = 'Dec deg per row pixel'

   # hdu.header["RCVAL1"] = float(self.cc[0])
   # hdu.header.comments["RCVAL1"] = 'Real center X of the data'
   # hdu.header["RCVAL2"] = float(self.cc[1])
   # hdu.header.comments["RCVAL2"] = 'Real center Y of the data'
   # hdu.header["RCVAL3"] = float(self.cc[2])
   # hdu.header.comments["RCVAL3"] = 'Real center Z of the data'
    hdu.header["UNITS"] = "kpc"
    hdu.header.comments["UNITS"] = 'Units for the RCVAL and PSIZE'
    hdu.header["PIXVAL"] = "mass *h^{-1}*10^{10}"
    hdu.header.comments["PIXVAL"] = 'The y parameter for thermal SZ effect.'
   # hdu.header["ORAD"] = float(self.rr)
   # hdu.header.comments["ORAD"] = 'Rcut in physical for the image.'
   # hdu.header["REDSHIFT"] = float(self.red)
   # hdu.header.comments["REDSHIFT"] = 'The redshift of the object being put to'
   # hdu.header["PSIZE"] = float(self.pxs)
   # hdu.header.comments["PSIZE"] = 'The pixel size in physical at simulation time'

   # hdu.header["AGLRES"] = float(self.ar)
   # hdu.header.comments["AGLRES"] = '\'observation\' angular resolution in arcsec'

    hdu.header["ORIGIN"] = 'Software: DMML'
    hdu.header.comments["ORIGIN"] = 'to be continued'
    #hdu.header["VERSION"] = version.version  # get_property('__version__')
    #hdu.header.comments["VERSION"] = 'Version of the software'
    hdu.header["DATE-OBS"] = Time.now().tt.isot
    if isinstance(comments, type([])) or isinstance(comments, type(())):
        for j in range(len(comments)):
            hdu.header["COMMENT"+str(j+1)] = comments[j]
    elif isinstance(comments, type("")) or isinstance(comments, type('')):
        hdu.header["COMMENT"] = comments
    else:
        raise ValueError("Do not accept this comments type! Please use str or list")
    hdu.writeto(fname, overwrite=overwrite)

#------------
# Open Fits
#------------



datapath_xr = '/data1/users/weiguang/X-ray/ML2/'
datapath_sz = '/data1/users/weiguang/SZ/ML2/'
datapath_dm = '/home2/deandres/maps/'

def read_xr(lp,hid,RA):
    region = 'NewMDCLUSTER_{}/'.format(str(lp).zfill(4))
    s = str(hid)[:3]
    file = 'snap_{}-Athena-wfi-cl-{}-ra-{}-{}-{}.fits'.format(s,hid,RA[0],RA[1],RA[2])
    print(RA)
    data = fits.getdata(datapath_xr+region+file)
    return data

def read_sz(lp,hid,RA):
    region = 'NewMDCLUSTER_{}/'.format(str(lp).zfill(4))
    s = str(hid)[:3]
    file = 'snap_{}-TT-cl-{}-ra-{}-{}-{}.fits'.format(s,hid,RA[0],RA[1],RA[2])
    print(RA)
    data = fits.getdata(datapath_sz+region+file)
    return data

def read_dm(lp,hid,RA):
    region = 'NewMDCLUSTER_{}/'.format(str(lp).zfill(4))
    s = str(hid)[:3]
    file = 'snap_{}-DM-cl-{}-ra-{}-{}-{}.fits'.format(s,hid,RA[0],RA[1],RA[2])
    print(RA)
    data = fits.getdata(datapath_dm+region+file)
    return data

def rotation_matrix(axis):

    sa, ca = np.sin(axis[0] / 180. *
                    np.pi), np.cos(axis[0] / 180. * np.pi)
    sb, cb = np.sin(axis[1] / 180. *
                    np.pi), np.cos(axis[1] / 180. * np.pi)
    sg, cg = np.sin(axis[2] / 180. *
                    np.pi), np.cos(axis[2] / 180. * np.pi)
    Rxyz = np.array(
        [[cb * cg, cg * sa * sb - ca * sg, ca * cg * sb + sa * sg],
         [cb * sg, ca * cg + sa * sb * sg, ca * sb * sg - cg * sa],
         [-sb,     cb * sa,                ca * cb]], dtype=np.float64)

    return Rxyz

def c_mask(img,radius=None,center=None ):

    h, w =  img.shape

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def mass_profile(img):
    M = []
    for i in range(int(len(img)/2)):
        mask = c_mask(img,radius=i)
        M.append(img[mask].sum())
    return np.array(M)*1e10 # this factors to account for units M_{sun}h^{-1}
