"""Strategy families: each is a distinct structural hypothesis about session liquidity."""

FAMILIES = {
    # A: baseline anchor — Asia sweeps prior NY-late extremes, reversal (their strategy)
    "A_asia_ny_rev": dict(
        refs=[((19, 21), 1)], acts=[(22, 9), (22, 10)],
        modes=["rev"], triggers=["sweep"], confirms=[False],
        bufs=[0.75, 1.0, 1.25], atr_lens=[10], mults=[1.0],
        tps=[0.75, "opp"], exits=[8],
    ),
    # B: London grabs — Asia/Tokyo range or prior NY extremes swept at London open
    "B_london_rev": dict(
        refs=[((19, 21), 1), ((0, 5), 1), ((22, 4), 0)],
        acts=[(6, 9), (6, 10), (7, 10)],
        modes=["rev"], triggers=["sweep"], confirms=[False, True],
        bufs=[0.25, 0.5, 0.75], atr_lens=[10], mults=[1.0, 1.5],
        tps=[0.5, 0.75, "opp"], exits=[10, 11],
    ),
    # C: NY Judas — London morning extremes swept at NY open, fade into NY afternoon
    "C_ny_rev": dict(
        refs=[((7, 12), 0), ((7, 13), 0), ((6, 11), 0)],
        acts=[(12, 15), (13, 16), (13, 17)],
        modes=["rev"], triggers=["sweep"], confirms=[False, True],
        bufs=[0.25, 0.5], atr_lens=[10], mults=[1.0, 1.5],
        tps=[0.5, 0.75, 1.0], exits=[17, 19],
    ),
    # D: Asia compression breakout — early Asia range breaks, ride into London
    "D_asia_breakout": dict(
        refs=[((22, 3), 0), ((0, 4), 0)],
        acts=[(3, 8), (4, 9)],
        modes=["cont"], triggers=["sweep"], confirms=[False],
        bufs=[0.25, 0.5], atr_lens=[10], mults=[1.0, 1.5, 2.0],
        tps=[1.0, 1.5, 2.0], exits=[8, 10],
    ),
    # E: previous-day extreme fade — PDH/PDL are liquidity magnets
    "E_pd_extreme_rev": dict(
        refs=[((0, 23), 1)],
        acts=[(22, 9), (6, 10), (13, 16)],
        act_exits={0: [8], 1: [11], 2: [17]},
        modes=["rev"], triggers=["sweep"], confirms=[False, True],
        bufs=[0.5, 1.0, 1.5], atr_lens=[10], mults=[1.0, 1.5],
        tps=[0.75, 1.0, "opp"], exits=None,
    ),
    # F: NY trend continuation — close beyond London high/low, ride NY afternoon
    "F_ny_trend_cont": dict(
        refs=[((7, 13), 0), ((6, 12), 0)],
        acts=[(14, 17), (15, 18)],
        modes=["cont"], triggers=["close_beyond"], confirms=[False],
        bufs=[0.0], atr_lens=[10], mults=[1.5, 2.0, 2.5],
        tps=[1.0, 1.5, 2.0], exits=[20, 21],
    ),
    # G: Asia momentum continuation — Asia breaks prior NY extreme and keeps going
    "G_asia_cont": dict(
        refs=[((19, 21), 1), ((19, 22), 1)],
        acts=[(22, 9), (22, 10)],
        modes=["cont"], triggers=["sweep"], confirms=[False],
        bufs=[0.5, 1.0], atr_lens=[10], mults=[1.0, 1.5],
        tps=[1.0, 1.5], exits=[9, 10],
    ),
}


def expand(families=FAMILIES):
    """yield (family_name, spec_dict)"""
    import itertools
    for fam, f in families.items():
        for ref, off in f["refs"]:
            for act in f["acts"]:
                exits_list = f["exits"]
                if exits_list is None:
                    idx = f["acts"].index(act)
                    exits_list = f["act_exits"][idx]
                for mode, trig, conf, buf, alen, mult, tp, ex in itertools.product(
                        f["modes"], f["triggers"], f["confirms"],
                        f["bufs"], f["atr_lens"], f["mults"], f["tps"], exits_list):
                    if trig == "close_beyond" and buf != 0.0:
                        continue
                    spec = dict(ref_win=ref, ref_off=off, act_win=act, mode=mode,
                                trigger=trig, confirm=conf, buf=buf, atr_len=alen,
                                mult=mult, tp=tp, exit_hour=ex, max_hold=36)
                    yield fam, spec


def spec_id(fam, spec):
    tp = spec["tp"]
    return (f"{fam}|ref{spec['ref_win'][0]:02d}-{spec['ref_win'][1]:02d}o{spec['ref_off']}"
            f"|act{spec['act_win'][0]:02d}-{spec['act_win'][1]:02d}|{spec['mode'][:3]}"
            f"{'' if spec['trigger'] == 'sweep' else '_cb'}"
            f"{'_cf' if spec['confirm'] else ''}"
            f"|b{spec['buf']}|a{spec['atr_len']}|m{spec['mult']}"
            f"|tp{tp}|x{spec['exit_hour']:02d}")
