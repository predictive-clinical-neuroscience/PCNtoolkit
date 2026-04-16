from __future__ import annotations

import os

import pandas as pd

from pcntoolkit.dataio.norm_data import NormData


def load_fcon1000(save_path: str | None = None):
    """Download and save fcon dataset to specified path, or load it from there
    if it is already downloaded

    Parameters
    ----------
    save_path : str | None
        The path to save the dataset to, or load it from if it is already
        downloaded

    Returns
    -------
    NormData
        The loaded dataset as a NormData object"""
    if not save_path:
        save_path = os.path.join("pcntoolkit_resources", "data")
    os.makedirs(save_path, exist_ok=True)
    data_path = os.path.join(save_path, "fcon1000.csv")

    # If the dataset is not already downloaded, download it and save it to
    # the specified path
    if not os.path.exists(data_path):
        data = pd.read_csv(
            "https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/refs/heads/main/data/fcon1000.csv"
        )
        data.to_csv(data_path, index=False)
    else:
        data = pd.read_csv(data_path)

    # Define the variables
    sex_map = {0: "F", 1: "M"}
    data["sex"] = data["sex"].map(sex_map)
    subject_ids = "sub_id"
    covariates = ["age"]
    batch_effects = ["sex", "site"]
    response_vars = [
        "lh_G&S_frontomargin_thickness",
        "lh_G&S_occipital_inf_thickness",
        "lh_G&S_paracentral_thickness",
        "lh_G&S_subcentral_thickness",
        "lh_G&S_transv_frontopol_thickness",
        "lh_G&S_cingul-Ant_thickness",
        "lh_G&S_cingul-Mid-Ant_thickness",
        "lh_G&S_cingul-Mid-Post_thickness",
        "lh_G_cingul-Post-dorsal_thickness",
        "lh_G_cingul-Post-ventral_thickness",
        "lh_G_cuneus_thickness",
        "lh_G_front_inf-Opercular_thickness",
        "lh_G_front_inf-Orbital_thickness",
        "lh_G_front_inf-Triangul_thickness",
        "lh_G_front_middle_thickness",
        "lh_G_front_sup_thickness",
        "lh_G_Ins_lg&S_cent_ins_thickness",
        "lh_G_insular_short_thickness",
        "lh_G_occipital_middle_thickness",
        "lh_G_occipital_sup_thickness",
        "lh_G_oc-temp_lat-fusifor_thickness",
        "lh_G_oc-temp_med-Lingual_thickness",
        "lh_G_oc-temp_med-Parahip_thickness",
        "lh_G_orbital_thickness",
        "lh_G_pariet_inf-Angular_thickness",
        "lh_G_pariet_inf-Supramar_thickness",
        "lh_G_parietal_sup_thickness",
        "lh_G_postcentral_thickness",
        "lh_G_precentral_thickness",
        "lh_G_precuneus_thickness",
        "lh_G_rectus_thickness",
        "lh_G_subcallosal_thickness",
        "lh_G_temp_sup-G_T_transv_thickness",
        "lh_G_temp_sup-Lateral_thickness",
        "lh_G_temp_sup-Plan_polar_thickness",
        "lh_G_temp_sup-Plan_tempo_thickness",
        "lh_G_temporal_inf_thickness",
        "lh_G_temporal_middle_thickness",
        "lh_Lat_Fis-ant-Horizont_thickness",
        "lh_Lat_Fis-ant-Vertical_thickness",
        "lh_Lat_Fis-post_thickness",
        "lh_Pole_occipital_thickness",
        "lh_Pole_temporal_thickness",
        "lh_S_calcarine_thickness",
        "lh_S_central_thickness",
        "lh_S_cingul-Marginalis_thickness",
        "lh_S_circular_insula_ant_thickness",
        "lh_S_circular_insula_inf_thickness",
        "lh_S_circular_insula_sup_thickness",
        "lh_S_collat_transv_ant_thickness",
        "lh_S_collat_transv_post_thickness",
        "lh_S_front_inf_thickness",
        "lh_S_front_middle_thickness",
        "lh_S_front_sup_thickness",
        "lh_S_interm_prim-Jensen_thickness",
        "lh_S_intrapariet&P_trans_thickness",
        "lh_S_oc_middle&Lunatus_thickness",
        "lh_S_oc_sup&transversal_thickness",
        "lh_S_occipital_ant_thickness",
        "lh_S_oc-temp_lat_thickness",
        "lh_S_oc-temp_med&Lingual_thickness",
        "lh_S_orbital_lateral_thickness",
        "lh_S_orbital_med-olfact_thickness",
        "lh_S_orbital-H_Shaped_thickness",
        "lh_S_parieto_occipital_thickness",
        "lh_S_pericallosal_thickness",
        "lh_S_postcentral_thickness",
        "lh_S_precentral-inf-part_thickness",
        "lh_S_precentral-sup-part_thickness",
        "lh_S_suborbital_thickness",
        "lh_S_subparietal_thickness",
        "lh_S_temporal_inf_thickness",
        "lh_S_temporal_sup_thickness",
        "lh_S_temporal_transverse_thickness",
        "lh_MeanThickness_thickness",
        "BrainSegVolNotVent",
        "eTIV",
        "rh_G&S_frontomargin_thickness",
        "rh_G&S_occipital_inf_thickness",
        "rh_G&S_paracentral_thickness",
        "rh_G&S_subcentral_thickness",
        "rh_G&S_transv_frontopol_thickness",
        "rh_G&S_cingul-Ant_thickness",
        "rh_G&S_cingul-Mid-Ant_thickness",
        "rh_G&S_cingul-Mid-Post_thickness",
        "rh_G_cingul-Post-dorsal_thickness",
        "rh_G_cingul-Post-ventral_thickness",
        "rh_G_cuneus_thickness",
        "rh_G_front_inf-Opercular_thickness",
        "rh_G_front_inf-Orbital_thickness",
        "rh_G_front_inf-Triangul_thickness",
        "rh_G_front_middle_thickness",
        "rh_G_front_sup_thickness",
        "rh_G_Ins_lg&S_cent_ins_thickness",
        "rh_G_insular_short_thickness",
        "rh_G_occipital_middle_thickness",
        "rh_G_occipital_sup_thickness",
        "rh_G_oc-temp_lat-fusifor_thickness",
        "rh_G_oc-temp_med-Lingual_thickness",
        "rh_G_oc-temp_med-Parahip_thickness",
        "rh_G_orbital_thickness",
        "rh_G_pariet_inf-Angular_thickness",
        "rh_G_pariet_inf-Supramar_thickness",
        "rh_G_parietal_sup_thickness",
        "rh_G_postcentral_thickness",
        "rh_G_precentral_thickness",
        "rh_G_precuneus_thickness",
        "rh_G_rectus_thickness",
        "rh_G_subcallosal_thickness",
        "rh_G_temp_sup-G_T_transv_thickness",
        "rh_G_temp_sup-Lateral_thickness",
        "rh_G_temp_sup-Plan_polar_thickness",
        "rh_G_temp_sup-Plan_tempo_thickness",
        "rh_G_temporal_inf_thickness",
        "rh_G_temporal_middle_thickness",
        "rh_Lat_Fis-ant-Horizont_thickness",
        "rh_Lat_Fis-ant-Vertical_thickness",
        "rh_Lat_Fis-post_thickness",
        "rh_Pole_occipital_thickness",
        "rh_Pole_temporal_thickness",
        "rh_S_calcarine_thickness",
        "rh_S_central_thickness",
        "rh_S_cingul-Marginalis_thickness",
        "rh_S_circular_insula_ant_thickness",
        "rh_S_circular_insula_inf_thickness",
        "rh_S_circular_insula_sup_thickness",
        "rh_S_collat_transv_ant_thickness",
        "rh_S_collat_transv_post_thickness",
        "rh_S_front_inf_thickness",
        "rh_S_front_middle_thickness",
        "rh_S_front_sup_thickness",
        "rh_S_interm_prim-Jensen_thickness",
        "rh_S_intrapariet&P_trans_thickness",
        "rh_S_oc_middle&Lunatus_thickness",
        "rh_S_oc_sup&transversal_thickness",
        "rh_S_occipital_ant_thickness",
        "rh_S_oc-temp_lat_thickness",
        "rh_S_oc-temp_med&Lingual_thickness",
        "rh_S_orbital_lateral_thickness",
        "rh_S_orbital_med-olfact_thickness",
        "rh_S_orbital-H_Shaped_thickness",
        "rh_S_parieto_occipital_thickness",
        "rh_S_pericallosal_thickness",
        "rh_S_postcentral_thickness",
        "rh_S_precentral-inf-part_thickness",
        "rh_S_precentral-sup-part_thickness",
        "rh_S_suborbital_thickness",
        "rh_S_subparietal_thickness",
        "rh_S_temporal_inf_thickness",
        "rh_S_temporal_sup_thickness",
        "rh_S_temporal_transverse_thickness",
        "rh_MeanThickness_thickness",
        "Left-Lateral-Ventricle",
        "Left-Inf-Lat-Vent",
        "Left-Cerebellum-White-Matter",
        "Left-Cerebellum-Cortex",
        "Left-Thalamus-Proper",
        "Left-Caudate",
        "Left-Putamen",
        "Left-Pallidum",
        "3rd-Ventricle",
        "4th-Ventricle",
        "Brain-Stem",
        "Left-Hippocampus",
        "Left-Amygdala",
        "CSF",
        "Left-Accumbens-area",
        "Left-VentralDC",
        "Left-vessel",
        "Left-choroid-plexus",
        "Right-Lateral-Ventricle",
        "Right-Inf-Lat-Vent",
        "Right-Cerebellum-White-Matter",
        "Right-Cerebellum-Cortex",
        "Right-Thalamus-Proper",
        "Right-Caudate",
        "Right-Putamen",
        "Right-Pallidum",
        "Right-Hippocampus",
        "Right-Amygdala",
        "Right-Accumbens-area",
        "Right-VentralDC",
        "Right-vessel",
        "Right-choroid-plexus",
        "5th-Ventricle",
        "WM-hypointensities",
        "Left-WM-hypointensities",
        "Right-WM-hypointensities",
        "non-WM-hypointensities",
        "Left-non-WM-hypointensities",
        "Right-non-WM-hypointensities",
        "Optic-Chiasm",
        "CC_Posterior",
        "CC_Mid_Posterior",
        "CC_Central",
        "CC_Mid_Anterior",
        "CC_Anterior",
        "BrainSegVol",
        "BrainSegVolNotVentSurf",
        "lhCortexVol",
        "rhCortexVol",
        "CortexVol",
        "lhCerebralWhiteMatterVol",
        "rhCerebralWhiteMatterVol",
        "CerebralWhiteMatterVol",
        "SubCortGrayVol",
        "TotalGrayVol",
        "SupraTentorialVol",
        "SupraTentorialVolNotVent",
        "SupraTentorialVolNotVentVox",
        "MaskVol",
        "BrainSegVol-to-eTIV",
        "MaskVol-to-eTIV",
        "lhSurfaceHoles",
        "rhSurfaceHoles",
        "SurfaceHoles",
        "EstimatedTotalIntraCranialVol",
    ]
    norm_data = NormData.from_dataframe(
        name="fcon1000",
        dataframe=data,
        covariates=covariates,
        batch_effects=batch_effects,
        response_vars=response_vars,
        subject_ids=subject_ids,
        remove_Nan=True,
    )
    return norm_data


# NOTE: This dataset is not public
def load_lifespan_big(
        n_response_vars: int | None = None,
        n_largest_sites: int | None = None,
        n_subjects: int | None = None
) -> NormData:
    """
    Load the lifespan_big dataset, which is a large lifespan dataset with many sites.

    Parameters
    ----------
    n_response_vars : int | None
        If specified, only use the first n_response_vars response
        variables.
    n_largest_sites : int | None
        If specified, only keep data from the n_largest_sites largest
        sites.
    n_subjects : int | None
        If specified, randomly sample n_subjects subjects.

    Returns
    -------
    NormData
        The loaded dataset as a NormData object.
    """
    # Define the variables in the dataset
    subject_ids = ["participant_id"]
    covariates = ["age"]
    batch_effects = ["sex", "site"]

    # Define the dtypes for loading the dataset, to ensure that categorical
    # variables are loaded as strings and numerical variables as floats
    dtypes = {"participant_id": str, "group": str, "group2": str}
    for col in batch_effects:
        dtypes[col] = str
    for col in covariates:
        dtypes[col] = float

    # Load the lifespan dataset with 57116 subjects from the Braicharts paper:
    # https://doi.org/10.7554/eLife.72904
    data = pd.read_csv(
        "/project_cephfs/3022017.06/projects/stijdboe/Data/sairut_data/"
        "lifespan_big.csv", dtype=dtypes)

    # Drop rows where all values are NaN
    data = data.dropna(axis=0, how="all", inplace=False)
    # Drop columns where even if 1 value is NaN
    data = data.dropna(axis=1, how="any", inplace=False)

    data["sex"] = data["sex"].map(
        {"0.0": "Female", "1.0": "Male", "2.0": "Female"})
    data["site"] = data["site_ID"]

    # If requested, take only the n largest sites
    if n_largest_sites is not None:
        data = data[data["site_ID"].isin(
            data["site_ID"].value_counts().head(n_largest_sites).index)]

    # If requested, take only n subjects
    if n_subjects is not None:
        data = data.sample(n=n_subjects, replace=False)

    # Define response variables as all variables that
    # are not in subject_ids, covariates, batch_effects,
    # and that have variance > 0
    def is_response_var(col_name: str) -> bool:
        return (
            col_name not in subject_ids
            and col_name not in covariates
            and col_name not in batch_effects
            and not col_name.startswith("site_")
            and not col_name.startswith("group")
            and not col_name.startswith("race")
            and data[col_name].var() > 0
        )

    response_vars = [col for col in data.columns if is_response_var(col)]

    # If requested, take only n response variables
    if n_response_vars is not None:
        response_vars = response_vars[:n_response_vars]

    # Create NormData object
    norm_data = NormData.from_dataframe(
        name="lifespan_big",
        dataframe=data,
        covariates=covariates,
        batch_effects=batch_effects,
        response_vars=response_vars,
        subject_ids=subject_ids,
    )

    return norm_data
