import time
import itertools

from .data_structures import *
from .qallse import Qallse, Config1GeV, Config
from .qallse_mp import QallseMp, MpConfig
from .qallse_base import QallseBase
from .utils import curvature, angle_diff

from typing import Union
import pandas as pd
import queue


class DoubletConfig(MpConfig):

    #: Select parsing doublets: All doublets or Filtered doublets using qplet
    use_dblt_from_qplt = False#True

    #: Use only connected doublets for QUBO building
    use_connected_dblt = True

    #: Apply impact parameter selection for doublets
    do_d0z0cut_forDblt = True

    #: maximum curvature calculated from minimum pT and magnetic strength
    magnetic_strength = 2 # 2 Tesla 
    min_transverse_momentum = 0.75 # 0.12 # [GeV]
    tplet_max_curv = (0.3 * magnetic_strength) /  (min_transverse_momentum * 1E+3) # [1/mm]
    
    #: cost for one doublet
    qubo_bias_weight = 1.

    #: cost for conflicting doublet
    qubo_conflict_strength = 5.

    #: Profit from related-doublets
    weight_multiplier = -0.3 * 2

    #: tune penalty term (for developping)
    temporary_sf = 1.0

    #: scale of the difference between two curvatures for x-y penalty
    my_qplet_max_dcurv = 1E-4

    # parameters of penalty functions
    volayer_power = 2
    xy_relative_strength = 0.5
    xy_power = 2
    zr_power = 2

    #: longitudinal width of the luminous region. In trackml: 55mm
    beamspot_width = 55 / 2.0

    #For Low pT
    if False:
        tplet_max_drz = 0.2
        qplet_max_dcurv = 5E-4
        my_qplet_max_dcurv = 1E-3


class QallseDoublet(QallseMp):
    """ Doublet-based QUBO. Inherited QallseMp for the use of qplet filtering."""
    config: DoubletConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _is_invalid_doublet(self, dblet: Doublet) -> bool:
        ret = super()._is_invalid_doublet(dblet)

        #: Selection using d0 & z0
        if self.config.do_d0z0cut_forDblt:
            c = curvature( (0,0), dblet.h1.coord_2d, dblet.h2.coord_2d )
            if abs(c) > self.config.tplet_max_curv:
                return True

            z0 = abs(dblet.h2.z - (dblet.dz / dblet.dr) * dblet.h2.r)
            if z0 > self.config.beamspot_width:
                return True
            pass
        return ret

    def _get_base_config(self):
        return DoubletConfig()

    def _create_triplets(self):
        #: if using filtered-doublets, apply triplet creation
        if self.config.use_dblt_from_qplt:
            super()._create_triplets()

    def _create_quadruplets(self, register_qubo=True):
        #: if using filtered-doublets, apply quadruplet creation.
        #: if not, register doublets & hits.
        if self.config.use_dblt_from_qplt:
            super()._create_quadruplets(register_qubo)
        else:
            self.qubo_doublets.update(self.doublets)
            for d in self.doublets:
                self.qubo_hits.update(zip(d.hit_ids(), d.hits))

    def _compute_weight(self, dplet: Doublet) -> float:
        #: TODO: Track-length-based weight?
        distance = (dplet.h2.volayer - dplet.h1.volayer) ** self.config.volayer_power
        return self.config.qubo_bias_weight * distance

    def _compute_conflict_strength(self, d1: Doublet, d2: Doublet) -> float:
        return self.config.qubo_conflict_strength

    def _penalty_rz(self, dblt1: Doublet, dblt2: Doublet) -> float:
        #: penalty term for r-z plane
        #: when doublet pair belongs to the same track, return 0
        rz_angle_mid = math.atan2(dblt2.h1.z - dblt1.h2.z, dblt2.h1.r - dblt1.h2.r)
        drz1 = angle_diff(dblt1.rz_angle, rz_angle_mid)
        drz2 = angle_diff(dblt2.rz_angle, rz_angle_mid)
        penalty1 = drz1 / self.config.tplet_max_drz
        penalty2 = drz2 / self.config.tplet_max_drz
        return (penalty1 ** 2 + penalty2 ** 2) ** 0.5


    def _penalty_xy(self, dblt1: Doublet, dblt2: Doublet) -> float:
        #: penalty term for x-y plane
        #: when doublet pair belongs to the same track, return 0
        assert dblt1.h2.r < dblt2.h1.r
        curvature1 = curvature( dblt1.h1.coord_2d, dblt1.h2.coord_2d, dblt2.h1.coord_2d )
        curvature2 = curvature( dblt1.h1.coord_2d, dblt1.h2.coord_2d, dblt2.h2.coord_2d )

        #: Reject Zigzag pattern when track momentum is low
        #: TODO: hard coding should be modified
        if abs(curvature1) > 1E-5 and abs(curvature2) > 1E-5:
            if curvature1 * curvature2 < 0:
                return 10

        curvature1 = abs(curvature1)
        curvature2 = abs(curvature2)
        #: Reject low momentum tracks
        if ((curvature1 > self.config.tplet_max_curv) 
            or (curvature2 > self.config.tplet_max_curv)):
            return 10

        if curvature1 >= curvature2:
            return ( curvature1 - curvature2 ) / self.config.my_qplet_max_dcurv
        else:
            #: Allow momentum descreasing (p_inner > p_outer)
            return ( curvature2 - curvature1 ) / self.config.my_qplet_max_dcurv / 2

    def _compute_consistency(self, dblt1: Doublet, dblt2: Doublet) -> float:
        """
        Calculate a consistency of doublet pair.
        If doublet pair does not belong to the same track, return 0
        For now, not consider the distance of two doublets.
        """

        assert dblt1.h1.r < dblt2.h1.r
        assert dblt1.h2.r < dblt2.h2.r

        penalty_rz = self._penalty_rz(dblt1, dblt2) * self.config.temporary_sf
        penalty_xy = self._penalty_xy(dblt1, dblt2) * self.config.temporary_sf

        if penalty_rz > 1 or penalty_xy > 1:
            return 0

        xy_strength = 1 - penalty_rz ** self.config.rz_power
        rz_strength = 1 - penalty_xy ** self.config.xy_power

        numerator = self.config.weight_multiplier * (
                self.config.xy_relative_strength * xy_strength +
                (1 - self.config.xy_relative_strength) * rz_strength
        )

        #exceeding_volayer_span = dblt2.h2.volayer - dblt1.h1.volayer - 3
        #denominator = (1 + exceeding_volayer_span) ** self.config.volayer_power
        denominator = 1

        strength = numerator / denominator

        # clip the strength if needed
        if self.config.strength_bounds is not None:
            strength = np.clip(strength, *self.config.strength_bounds)

        return strength

    def to_qubo(self, return_stats=False) -> Union[TQubo, Tuple[TQubo, Tuple[int, int, int]]]:
        """
        Generate the doublet-based QUBO. 
        """

        Q = {}
        hits, doublets, triplets = self.qubo_hits, self.qubo_doublets, self.qubo_triplets
        quadruplets = self.quadruplets

        if not self.config.use_dblt_from_qplt:
            for hit in hits.values():
                hit.outer_kept.update(hit.outer)
                hit.inner_kept.update(hit.inner)
            for d in doublets:
                d.h1.outer_kept.add(d)
                d.h2.inner_kept.add(d)

        start_time = time.process_time()
        print (self.config.as_dict())

        # 1: qbits with their weight (doublets with a common weight)
        for d in doublets:
            Q[(str(d), str(d))] = self._compute_weight(d)
        n_vars = len(Q)

        for d1 in doublets:
            assert d1.h1.r < d1.h2.r

            if self.config.use_connected_dblt:
                #: Use only connected doublets
                #: only checking outer doublets to avoid double-counting
                set_dblt = set()
                que_dblt = queue.Queue()
                for d_out in d1.h2.outer_kept:
                    set_dblt.add(d_out)
                    que_dblt.put(d_out)

                while not que_dblt.empty():
                    d = que_dblt.get()
                    for d_out in d.h2.outer_kept:
                        if not d_out in set_dblt:
                            set_dblt.add(d_out)
                            que_dblt.put(d_out)
                doublets2 = set_dblt
            else:
                # when using all doublets
                doublets2 = doublets

            for d2 in doublets2:
                if d2.h1.r >= d2.h2.r: continue # not necessary when using only connected doublets
                if str(d1) == str(d2): continue # not necessary when using only connected doublets
                if d1.h1.r >= d2.h1.r: continue # not necessary when using only connected doublets
                if d1.h2.r > d2.h1.r: continue # not necessary when using only connected doublets
                assert d2.h1.r < d2.h2.r
                assert str(d1) != str(d2)
                assert d1.h1.r < d2.h1.r
                assert d1.h2.r <= d2.h1.r

                key = (str(d1), str(d2))
                if key in Q or tuple(reversed(key)) in Q:
                    continue
                nHits = len(set([d1.h1.volayer, d1.h2.volayer, d2.h1.volayer, d2.h2.volayer]))
                if nHits == 4:
                    #: inclusion couplers
                    weight = self._compute_consistency(d1, d2)
                    if weight < -1e-2:
                        distance = 1 # TODO: include distance?
                        Q[key] = weight / distance
                elif nHits == 3:
                    #: TODO: add proper inclusion couplers
                    if (d1.h2.hit_id == d2.h1.hit_id):
                        if False:
                            #: For now, turn off this penalty term
                            dzr = angle_diff(d1.rz_angle, d2.rz_angle)
                            sigma = (dzr / self.config.tplet_max_drz) * self.config.temporary_sf
                            Q[key] = 1 - math.exp(-sigma**2/2)

                        if False:
                            #: reject low momentum triplets
                            p0 = [d1.h1.x, d1.h1.y]
                            p1 = [d1.h2.x, d1.h2.y]
                            p2 = [d2.h2.x, d2.h2.y]
                            c = abs( curvature( p0, p1, p2 ) )
                            if c > self.config.tplet_max_curv:
                                Q[key] = self._compute_conflict_strength(d1, d2)

        n_incl_couplers = len(Q) - (n_vars)

        #: exclusion couplers (no two doublets can share the same hit)
        for _, hit in hits.items():
            for conflicts in [hit.inner_kept, hit.outer_kept]:
                for (d1, d2) in itertools.combinations(conflicts, 2):
                    assert d1 != d2
                    key = (str(d1), str(d2))
                    if key not in Q and tuple(reversed(key)) not in Q:
                        Q[key] = self._compute_conflict_strength(d1, d2)
        n_excl_couplers = len(Q) - (n_vars + n_incl_couplers)

        exec_time = time.process_time() - start_time

        #print (Q)

        self.logger.info(f'Qubo generated in {exec_time:.2f}s. Size: {len(Q)}. Vars: {n_vars}, '
                         f'excl. couplers: {n_excl_couplers}, incl. couplers: {n_incl_couplers}')
        if return_stats:
            return Q, (n_vars, n_incl_couplers, n_excl_couplers)
        else:
            return Q
