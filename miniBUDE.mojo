from std.sys import has_accelerator, argv
from std.collections import List
from std.math import sin, cos, sqrt
from std.time import monotonic
from std.utils.numerics import max_finite

from mojito import Mojito, array, array_ref

comptime NUM_ITER = 100
comptime NUM_POSES = 65536
comptime WG_SIZE = 64      # Work group size
# DEFAULT_PPWI 1, 2, 4, 8, 16, 32, 64, 128
comptime PPWI = 4          # Poses per work item
comptime DIFF_TOLERANCE_PCT = 0.025

comptime Zero = 0.0
comptime Quarter = 0.25
comptime Half = 0.5
comptime One = 1.0
comptime Two = 2.0
comptime Four = 4.0
comptime Cnstnt = 45.0

comptime HBTYPE_F = 70
comptime HBTYPE_E = 69
comptime HARDNESS = 38.0
comptime NPNPDIST = 5.5
comptime NPPDIST = 1.0

comptime dtype = DType.float32
comptime FloatMax = max_finite[dtype]()

comptime NUM_WITEM = NUM_POSES // PPWI

# Input-deck-specific
# Values for bm1 deck:
comptime NATPRO = 938
comptime NATLIG = 26
comptime NFF = 34

# Values for bm2 deck:
# comptime NATPRO = 2672
# comptime NATLIG = 2672
# comptime NFF = 44

struct Vec3f32(ImplicitlyCopyable, Movable):
    var x: Float32
    var y: Float32
    var z: Float32

    def __init__(out self, x: Float32, y: Float32, z: Float32):
        self.x = x
        self.y = y
        self.z = z

struct Vec4f32(ImplicitlyCopyable, Movable):
    var x: Float32
    var y: Float32
    var z: Float32
    var w: Float32

    def __init__(out self, x: Float32, y: Float32, z: Float32, w: Float32):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

struct Atom(ImplicitlyCopyable, Movable):
    var x: Float32
    var y: Float32
    var z: Float32
    var type: Int32

    def __init__(out self, x: Float32, y: Float32, z: Float32, type: Int32):
        self.x = x
        self.y = y
        self.z = z
        self.type = type

    @staticmethod
    def read_atoms(path: String) raises -> List[Atom]:
        var file = open(path, "r")
        var bytes = file.read_bytes()
        file.close()

        var ptr = bytes.unsafe_ptr().bitcast[UInt8]()
        var atoms = List[Atom]()
        var atom_size = 16      # Float32 x,y,z + Int32 type = 16 bytes

        var byte_count = len(bytes)
        var total = byte_count // atom_size

        for i in range(total):
            var offset = i * atom_size
            var x = (ptr + offset +  0).bitcast[Float32]()[0]
            var y = (ptr + offset +  4).bitcast[Float32]()[0]
            var z = (ptr + offset +  8).bitcast[Float32]()[0]
            var t = (ptr + offset + 12).bitcast[Int32]()[0]
            atoms.append(Atom(x, y, z, t))
        return atoms^

struct FFParams(ImplicitlyCopyable, Movable):
    var hbtype: Int32
    var radius: Float32
    var hphb: Float32
    var elsc: Float32

    def __init__(out self, hbtype: Int32, radius: Float32, hphb: Float32, elsc: Float32):
        self.hbtype = hbtype
        self.radius = radius
        self.hphb = hphb
        self.elsc = elsc

    @staticmethod
    def read_ffparams(path: String) raises -> List[FFParams]:
        var file = open(path, "r")
        var bytes = file.read_bytes()
        file.close()

        var ptr = bytes.unsafe_ptr().bitcast[UInt8]()
        var ffparams = List[FFParams]()
        var atom_size = 16      # 3 Float32s + 1 Int = 16 bytes

        var byte_count = len(bytes)
        var total = byte_count // atom_size

        for i in range(total):
            var offset = i * atom_size
            var hbtype = (ptr + offset +  0).bitcast[Int32]()[0]
            var radius = (ptr + offset +  4).bitcast[Float32]()[0]
            var hphb   = (ptr + offset +  8).bitcast[Float32]()[0]
            var elsc   = (ptr + offset + 12).bitcast[Float32]()[0]
            ffparams.append(FFParams(hbtype, radius, hphb, elsc))
        return ffparams^

def read_poses(path: String) raises -> List[List[Float32]]:
    var file = open(path, "r")
    var bytes = file.read_bytes()
    file.close()

    var ptr = bytes.unsafe_ptr().bitcast[Float32]()
    var total_floats = len(bytes) // 4
    if total_floats % 6 != 0:
        raise Error("Pose size (", total_floats, ") not divisible by 6")

    var num_poses = total_floats // 6
    if not num_poses == NUM_POSES:
        raise Error("Number of poses", num_poses, "doesn't match the expected:", NUM_POSES)

    var poses = List[List[Float32]]()
    for i in range(6):
        var component = List[Float32](capacity=NUM_POSES)
        for j in range(NUM_POSES):
            component.append(ptr[i * num_poses + j])
        poses.append(component^)
    return poses^

struct Params:
    var num_poses: Int
    var iterations: Int
    var wgsize: Int
    var ppwi: Int
    var deck: String

    def __init__(out self,
                num_poses: Int = NUM_POSES,
                iterations: Int = NUM_ITER,
                wgsize: Int = WG_SIZE,
                ppwi: Int = PPWI,
                deck: String = "minibude_data/bm1"):
        self.num_poses = num_poses
        self.iterations = iterations
        self.wgsize = wgsize
        self.ppwi = ppwi
        self.deck = deck

@fieldwise_init
struct Deck:
    var protein: List[Atom]
    var ligand: List[Atom]
    var forcefield: List[FFParams]
    var poses: List[List[Float32]]

def fasten_body(
    idx: Int,
    protein: array_ref[dtype, 4, NATPRO],
    ligand: array_ref[dtype, 4, NATLIG],
    forcefield: array_ref[dtype, 4, NFF],
    transforms: array_ref[dtype, 6, NUM_POSES],
    etotals: array_ref[dtype, NUM_POSES],
) -> None:
    # Compute transformation matrix to private memory
    var etot = InlineArray[Float32, PPWI](fill=0)
    var transform = InlineArray[Vec4f32, PPWI * 3](uninitialized=True)

    for i in range(PPWI):
        var index = idx * PPWI + i

        var sx: Float32 = sin(transforms[0, index])
        var cx: Float32 = cos(transforms[0, index])
        var sy: Float32 = sin(transforms[1, index])
        var cy: Float32 = cos(transforms[1, index])
        var sz: Float32 = sin(transforms[2, index])
        var cz: Float32 = cos(transforms[2, index])

        transform[i * 3].x = cy * cz
        transform[i * 3].y = sx * sy * cz - cx * sz
        transform[i * 3].z = cx * sy * cz + sx * sz
        transform[i * 3].w = transforms[3, index]
        transform[i * 3 + 1].x = cy * sz
        transform[i * 3 + 1].y = sx * sy * sz + cx * cz
        transform[i * 3 + 1].z = cx * sy * sz - sx * cz
        transform[i * 3 + 1].w = transforms[4, index]
        transform[i * 3 + 2].x = -sy
        transform[i * 3 + 2].y = sx * cy
        transform[i * 3 + 2].z = cx * cy
        transform[i * 3 + 2].w = transforms[5, index]

        etot[i] = Zero

    # Loop over ligand atoms
    var il = 0
    while True:
        var l_atom = Atom(ligand[0, il], ligand[1, il], ligand[2, il], Int32(ligand[3, il]))
        var l_params = FFParams(Int32(forcefield[0, Int(l_atom.type)]), forcefield[1, Int(l_atom.type)], forcefield[2, Int(l_atom.type)], forcefield[3, Int(l_atom.type)])
        var lhphb_ltz = l_params.hphb < Zero
        var lhphb_gtz = l_params.hphb > Zero

        var lpos = InlineArray[Vec3f32, PPWI](uninitialized=True)
        var linitpos = Vec4f32(l_atom.x, l_atom.y, l_atom.z, One)
        for i in range(PPWI):
            var t0 = transform[i * 3]
            var t1 = transform[i * 3 + 1]
            var t2 = transform[i * 3 + 2]
            lpos[i].x = t0.w + linitpos.x * t0.x + linitpos.y * t0.y + linitpos.z * t0.z
            lpos[i].y = t1.w + linitpos.x * t1.x + linitpos.y * t1.y + linitpos.z * t1.z
            lpos[i].z = t2.w + linitpos.x * t2.x + linitpos.y * t2.y + linitpos.z * t2.z

        # Loop over protein atoms
        var ip = 0
        while True:
            var p_atom = Atom(protein[0, ip], protein[1, ip], protein[2, ip], Int32(protein[3, ip]))
            var p_params = FFParams(Int32(forcefield[0, Int(p_atom.type)]), forcefield[1, Int(p_atom.type)], forcefield[2, Int(p_atom.type)], forcefield[3, Int(p_atom.type)])

            var radij = p_params.radius + l_params.radius
            var r_radij = 1.0 / radij

            var elcdst: Float32
            if p_params.hbtype == HBTYPE_F and l_params.hbtype == HBTYPE_F:
                elcdst = Four
            else:
                elcdst = Two

            var elcdst1: Float32
            if p_params.hbtype == HBTYPE_F and l_params.hbtype == HBTYPE_F:
                elcdst1 = Quarter
            else:
                elcdst1 = Half

            var type_E    = p_params.hbtype == HBTYPE_E or l_params.hbtype == HBTYPE_E
            var phphb_ltz = p_params.hphb < Zero
            var phphb_gtz = p_params.hphb > Zero
            var phphb_nz  = p_params.hphb != Zero

            var p_hphb = p_params.hphb
            if phphb_ltz and lhphb_gtz:
                p_hphb *= -One
            else:
                p_hphb *= One

            var l_hphb = l_params.hphb
            if phphb_gtz and lhphb_ltz:
                l_hphb *= -One
            else:
                l_hphb *= One

            var distdslv: Float32
            if phphb_ltz:
                if lhphb_ltz:
                    distdslv = NPNPDIST
                else:
                    distdslv = NPPDIST
            else:
                if lhphb_ltz:
                    distdslv = NPPDIST
                else:
                    distdslv = -FloatMax

            var r_distdslv = 1.0 / distdslv
            var chrg_init  = l_params.elsc * p_params.elsc
            var dslv_init  = p_hphb + l_hphb

            for i in range(PPWI):
                var x      = lpos[i].x - p_atom.x
                var y      = lpos[i].y - p_atom.y
                var z      = lpos[i].z - p_atom.z
                var distij = sqrt(x * x + y * y + z * z)
                var distbb = distij - radij
                var zone1  = distbb < Zero

                # Calculate steric energy
                var tmp = One - distij * r_radij
                if zone1:
                    tmp *= Two * HARDNESS
                else:
                    tmp *= Zero
                etot[i] += tmp

                # Calculate formal and dipole charge interactions
                var f1: Float32
                if zone1:
                    f1 = One
                else:
                    f1 = One - distbb * elcdst1
                var f2: Float32
                if distbb < elcdst:
                    f2 = One
                else:
                    f2 = Zero
                var chrg_e     = chrg_init * f1 * f2
                var neg_chrg_e = -abs(chrg_e)
                if type_E:
                    chrg_e = neg_chrg_e
                else:
                    chrg_e = chrg_e
                etot[i] += chrg_e * Cnstnt

                # Calculate the two cases for Nonpolar-Polar repulsive interactions
                var coeff  = One - distbb * r_distdslv
                var dslv_e = dslv_init
                if distbb < distdslv and phphb_nz:
                    dslv_e *= One
                else:
                    dslv_e *= Zero
                if zone1:
                    dslv_e *= One
                else:
                    dslv_e *= coeff
                etot[i] += dslv_e

            ip += 1
            if ip >= NATPRO:
                break
        il += 1
        if il >= NATLIG:
            break

    # Write results
    for i in range(PPWI):
        etotals[idx * PPWI + i] = etot[i] * Half

def fill_protein[b: String](mut a: array[b, dtype, 4, NATPRO], deck: Deck) raises:
    for i in range(NATPRO):
        a[0, i] = deck.protein[i].x
        a[1, i] = deck.protein[i].y
        a[2, i] = deck.protein[i].z
        a[3, i] = Float32(deck.protein[i].type)

def fill_ligand[b: String](mut a: array[b, dtype, 4, NATLIG], deck: Deck) raises:
    for i in range(NATLIG):
        a[0, i] = deck.ligand[i].x
        a[1, i] = deck.ligand[i].y
        a[2, i] = deck.ligand[i].z
        a[3, i] = Float32(deck.ligand[i].type)

def fill_forcefield[b: String](mut a: array[b, dtype, 4, NFF], deck: Deck) raises:
    for i in range(NFF):
        a[0, i] = Float32(deck.forcefield[i].hbtype)
        a[1, i] = deck.forcefield[i].radius
        a[2, i] = deck.forcefield[i].hphb
        a[3, i] = deck.forcefield[i].elsc

def fill_transforms[b: String](mut a: array[b, dtype, 6, NUM_POSES], deck: Deck) raises:
    for comp in range(6):
        for j in range(NUM_POSES):
            a[comp, j] = deck.poses[comp][j]

def run[backend: String]() raises:
    var args = argv()
    var csv_output = False

    var i = 0
    while i < len(args):
        var arg = args[i]
        if arg == "--csv":
            csv_output = True
        i += 1

    var params = Params()
    var protein    = Atom.read_atoms(params.deck + "/protein.in")
    var ligand     = Atom.read_atoms(params.deck + "/ligand.in")
    var forcefield = FFParams.read_ffparams(params.deck + "/forcefield.in")
    var poses      = read_poses(params.deck + "/poses.in")
    var deck       = Deck(protein^, ligand^, forcefield^, poses^)

    var mj = Mojito[backend]()

    if not csv_output:
        print("Backend   :", backend)
        print("Poses     : ", len(deck.poses[0]))
        print("Iterations: ", params.iterations)
        print("Ligands   : ", len(deck.ligand))
        print("Protein   : ", len(deck.protein))
        print("Forcefield: ", len(deck.forcefield))
        print("Deck      : ", params.deck)
        print("WGsize    : ", params.wgsize)
        print("PPWI      : ", params.ppwi)
        print("")

    var etotals_arr    = mj.zeros[dtype, NUM_POSES]()
    var protein_arr    = mj.empty[dtype, 4, NATPRO]()
    var ligand_arr     = mj.empty[dtype, 4, NATLIG]()
    var forcefield_arr = mj.empty[dtype, 4, NFF]()
    var transforms_arr = mj.empty[dtype, 6, NUM_POSES]()

    protein_arr.to_host()
    ligand_arr.to_host()
    forcefield_arr.to_host()
    transforms_arr.to_host()
    mj.sync()

    fill_protein[backend](protein_arr, deck)
    fill_ligand[backend](ligand_arr, deck)
    fill_forcefield[backend](forcefield_arr, deck)
    fill_transforms[backend](transforms_arr, deck)

    protein_arr.to_device()
    ligand_arr.to_device()
    forcefield_arr.to_device()
    transforms_arr.to_device()

    # Warmup
    mj.parallel_for[NUM_WITEM, func=fasten_body](
        protein_arr, ligand_arr, forcefield_arr, transforms_arr, etotals_arr)

    # Timing 
    var kernel_times = List[Float64]()
    var total_elapsed: Float64 = 0.0

    for _ in range(NUM_ITER):
        var start = monotonic()
        mj.parallel_for[NUM_WITEM, func=fasten_body](
            protein_arr, ligand_arr, forcefield_arr, transforms_arr, etotals_arr)
        var end = monotonic()
        var elapsed = Float64(end - start)
        kernel_times.append(elapsed)
        total_elapsed += elapsed

    # Validate results
    etotals_arr.to_host()
    mj.sync()

    # Load reference energies
    var ref_energies = List[Float32]()
    var ref_file = open(params.deck + "/ref_energies.out", "r")
    var ref_content = ref_file.read()
    ref_file.close()
    var ref_lines = ref_content.split("\n")
    for j in range(len(ref_lines)):
        if len(ref_energies) >= params.num_poses:
            break
        var line = ref_lines[j].strip()
        if line.byte_length() > 0:
            ref_energies.append(Float32(atof(line)))

    # Verify correctness
    var max_diff_pct: Float64 = 0.0
    var num_failed: Int = 0
    for i in range(params.num_poses):
        var ref_val = Float64(ref_energies[i])
        var com_val = Float64(etotals_arr[i])
        # don't verify anything less than one
        if abs(ref_val) < 1.0 and abs(com_val) < 1.0:
            continue
        var diff_pct = abs(ref_val - com_val) / abs(ref_val) * 100.0
        if diff_pct > max_diff_pct:
            max_diff_pct = diff_pct
        if diff_pct > DIFF_TOLERANCE_PCT:
            num_failed += 1

    var valid = max_diff_pct < DIFF_TOLERANCE_PCT
    if not csv_output:
        if valid:
            print("Validation: PASS (max_diff_%:", max_diff_pct, ")")
        else:
            print("Validation: FAIL (max_diff_%:", max_diff_pct,
                  "failed:", num_failed, "/", params.num_poses, ")")

    if csv_output:
        # print("backend,GPU,ppwi,wgsize,sum_ms,avg_ms,min_ms,max_ms,stddev_ms,gflops/s")

        # Average time per iteration
        var ns      = total_elapsed / Float64(NUM_ITER)
        var runtime = ns * 1e-9

        # Compute FLOP/s
        var ops_per_wg = UInt32(PPWI * 27 + len(deck.ligand) * (2 + PPWI * 18 + len(deck.protein) * (10 + PPWI * 30)) + PPWI)
        var total_ops = Float64(ops_per_wg) * (Float64(NUM_POSES) / Float64(PPWI))
        var flops  = total_ops / runtime
        var gflops = flops / 1e9

        # Compute timing stats in ms
        var sum_ms: Float64 = 0.0
        var min_ms: Float64 = kernel_times[0] * 1e-6
        var max_ms: Float64 = kernel_times[0] * 1e-6
        for i in range(NUM_ITER):
            var t_ms = kernel_times[i] * 1e-6
            sum_ms += t_ms
            if t_ms < min_ms:
                min_ms = t_ms
            if t_ms > max_ms:
                max_ms = t_ms
        var avg_ms = sum_ms / Float64(NUM_ITER)
        var variance: Float64 = 0.0
        for i in range(NUM_ITER):
            var d = kernel_times[i] * 1e-6 - avg_ms
            variance += d * d
        variance /= Float64(NUM_ITER)
        var stddev_ms = sqrt(variance)

        print("Mojo,", backend, ",", PPWI, ",", WG_SIZE, ",",
              sum_ms, ",", avg_ms, ",", min_ms, ",", max_ms, ",",
              stddev_ms, ",", gflops)

def main() raises:
    run["gpu"]()
    run["cpu"]()
