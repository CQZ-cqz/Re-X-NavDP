"""Convert GRScenes MDL materials to UsdPreviewSurface (flat-color) materials.

Isaac Sim 5.1.0 deadlocks in neuraylib when compiling the external ``KooPbr``
MDL materials that the InternScenes home/commercial scenes reference. This
script rewrites the material shaders in the shared ``models/*/instance.usd``
files so they use the built-in ``UsdPreviewSurface`` instead, which renders
without any MDL compilation.

The conversion keeps the flat base colors (``BaseColor_Color``, ``Metallic_Color``,
``Gloss_Color``) and drops the texture maps, so appearance is a color-only
approximation but geometry and navigation metadata are untouched.

Usage (one model instance):
    python ../tools/assets/convert_mdl_to_previewsurface.py <instance.usd>

Batch (dry-run first, then apply):
    find .../models -name instance.usd -print0 | xargs -0 -P8 \
        python ../tools/assets/convert_mdl_to_previewsurface.py

Each file is rewritten in place; pass --backup to keep a .bak copy.
"""
import argparse
import os
import shutil
import sys

# Make the bundled pxr Python bindings importable without launching Isaac Sim.
_PXR_ROOT = None
for _cand in (
    "/home/cqz/.conda/envs/navrl/lib/python3.11/site-packages/isaacsim/extscache",
):
    if os.path.isdir(_cand):
        _PXR_ROOT = _cand
        break

if _PXR_ROOT is not None:
    _usd_libs = None
    for _d in os.listdir(_PXR_ROOT):
        if _d.startswith("omni.usd.libs-"):
            _usd_libs = os.path.join(_PXR_ROOT, _d)
            break
    if _usd_libs:
        sys.path.insert(0, _usd_libs)
        os.environ.setdefault("LD_LIBRARY_PATH", "")
        for _sub in ("bin", "bin/deps"):
            _p = os.path.join(_usd_libs, _sub)
            if _p not in os.environ["LD_LIBRARY_PATH"]:
                os.environ["LD_LIBRARY_PATH"] = (
                    f"{_p}:" + os.environ["LD_LIBRARY_PATH"]
                ).rstrip(":")

from pxr import Gf, Sdf, Usd, UsdShade  # noqa: E402

import re  # noqa: E402


def _parse_mdl_diffuse(mdl_path: str):
    """Extract the flat ``diffuse: color(r,g,b)`` from a local KooPbr .mdl file."""
    try:
        with open(mdl_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        m = re.search(r"diffuse:\s*color\(([^)]+)\)", content)
        if m:
            vals = [float(x.strip().rstrip("f")) for x in m.group(1).split(",")[:3]]
            return Gf.Vec3f(*vals)
    except OSError:
        pass
    return None


def _parse_mdl_texture(mdl_path: str):
    """Extract the first ``texture_2d("./textures/...")`` path from a local .mdl file."""
    try:
        with open(mdl_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        m = re.search(r'texture_2d\("([^"]+)"', content)
        if m:
            return m.group(1)  # e.g. "./textures/a9a/72e/....jpg"
    except OSError:
        pass
    return None


def _is_default_texture(tex_path: str) -> bool:
    """True for the white/black/normal placeholder textures used by flat materials."""
    name = str(tex_path).lower().rstrip("/").split("/")[-1]
    return name in ("white.png", "black.png", "normal.png", "white.jpg", "black.jpg")

MDL_INPUTS = (
    "BaseColor_Color", "BaseColor_Tex", "BaseColor_UVA",
    "Metallic_Color", "Metallic_Tex", "Metallic_UVA",
    "Gloss_Color", "Gloss_Tex", "Gloss_UVA",
    "Specular_Color", "Specular_Tex", "Specular_UVA",
    "Normal_Tex", "Normal_UVA",
    "Emissive_Color", "Emissive_Tex", "Emissive_UVA", "EmissiveIntensity",
    "IsBaseColorTex", "IsMetallicTex", "IsGlossTex", "IsSpecularTex", "IsEmissiveTex",
    "MaxTexCoordIndex", "PolygonOffset",
)


def _vec3(v, default=(0.8, 0.8, 0.8)):
    if v is None:
        return Gf.Vec3f(*default)
    return Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))


def convert_stage(stage: Usd.Stage, path: str) -> int:
    """Rewrite every MDL shader (except the HDR environment light) to UsdPreviewSurface.

    KooPbr materials keep their base color, and textured materials get their texture
    map re-wired through a ``UsdUVTexture`` shader (so appearance is preserved). The
    color/texture source is:
      * ``Num*`` shaders -> ``inputs:IsBaseColorTex`` / ``inputs:BaseColor_Tex``.
      * ``MI_*`` shaders (no inputs) -> parsed from the local .mdl source.
    ``WorldGridMaterial`` and remote MDL URLs fall back to a neutral gray.
    ``DayMaterial`` (the HDR image-based-light) is left untouched.
    """
    converted = 0
    for prim in list(stage.Traverse()):
        if not prim.IsA(UsdShade.Shader):
            continue
        mdl_asset = prim.GetAttribute("info:mdl:sourceAsset")
        if not mdl_asset or not mdl_asset.Get():
            continue
        if "DayMaterial" in str(mdl_asset.Get()):
            continue  # keep the HDR environment light as MDL

        parent = prim.GetParent()
        base_color = prim.GetAttribute("inputs:BaseColor_Color").Get()
        metallic = prim.GetAttribute("inputs:Metallic_Color").Get()
        gloss = prim.GetAttribute("inputs:Gloss_Color").Get()
        is_base_tex = prim.GetAttribute("inputs:IsBaseColorTex").Get()
        base_tex = prim.GetAttribute("inputs:BaseColor_Tex").Get()

        diffuse_texture = None  # relative texture path for UsdUVTexture, else flat color

        # Num* shaders: texture flag lives in the shader inputs.
        if is_base_tex is not None and float(is_base_tex) > 0.5 and base_tex is not None:
            t = str(base_tex).strip("@")
            if not _is_default_texture(t):
                diffuse_texture = t

        # MI_* shaders (no BaseColor inputs): color/texture lives in the .mdl source.
        if diffuse_texture is None and base_color is None:
            asset_str = str(mdl_asset.Get()).strip("@")
            local_name = asset_str.split("/")[-1]
            local_mdl = os.path.join(os.path.dirname(path), "Materials", local_name)
            if local_name.endswith(".mdl"):
                tex = _parse_mdl_texture(local_mdl)
                if tex:
                    diffuse_texture = "./Materials/" + tex.lstrip("./")
                else:
                    parsed = _parse_mdl_diffuse(local_mdl)
                    base_color = (
                        Gf.Vec4f(parsed[0], parsed[1], parsed[2], 1.0)
                        if parsed is not None
                        else Gf.Vec4f(0.5, 0.5, 0.5, 1.0)
                    )

        # Re-identify as the built-in UsdPreviewSurface shader.
        prim.GetAttribute("info:id").Set("UsdPreviewSurface")
        prim.GetAttribute("info:implementationSource").Set("id")
        prim.RemoveProperty("info:mdl:sourceAsset")
        prim.RemoveProperty("info:mdl:sourceAsset:subIdentifier")

        # Drop all MDL-specific inputs.
        for name in MDL_INPUTS:
            prim.RemoveProperty(f"inputs:{name}")

        # Add UsdPreviewSurface inputs.
        shader = UsdShade.Shader(prim)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            float(metallic[0]) if metallic is not None else 0.0
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            1.0 - (float(gloss[0]) if gloss is not None else 0.5)
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)

        if diffuse_texture is not None:
            # Wire the diffuse color through a UsdUVTexture so the original texture map shows.
            tex_prim_path = parent.GetPath().AppendChild("diffuseTex")
            tex_shader = UsdShade.Shader.Define(stage, tex_prim_path)
            tex_shader.CreateIdAttr("UsdUVTexture")
            tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(diffuse_texture)
            diffuse_in = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            diffuse_in.ConnectToSource(tex_shader.ConnectableAPI(), "rgb")
        else:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(_vec3(base_color))

        # Rename the MDL token output "out" -> UsdPreviewSurface "surface".
        prim.RemoveProperty("outputs:out")
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)

        # Re-point the parent Material's surface output to the shader's surface.
        if parent is not None and parent.IsA(UsdShade.Material):
            parent.RemoveProperty("outputs:mdl:surface")
            parent.CreateAttribute("outputs:surface", Sdf.ValueTypeNames.Token).SetConnections(
                [prim.GetPath().AppendProperty("outputs:surface")]
            )

        converted += 1
    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--backup", action="store_true")
    parser.add_argument(
        "--load-none",
        action="store_true",
        help="Open with LoadNone (only convert prims defined in the root layer; "
        "use for scene USDs that reference model instance.usd files).",
    )
    args = parser.parse_args()

    total = 0
    for path in args.paths:
        if not os.path.isfile(path):
            print(f"skip (missing): {path}")
            continue
        if args.backup:
            shutil.copy2(path, path + ".bak")
        load = Usd.Stage.LoadNone if args.load_none else Usd.Stage.LoadAll
        stage = Usd.Stage.Open(path, load=load)
        n = convert_stage(stage, path)
        stage.Save()
        total += n
        print(f"{n:3d}  {path}")
    print(f"TOTAL converted shaders: {total}")


if __name__ == "__main__":
    main()
