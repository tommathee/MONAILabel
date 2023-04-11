import xml.etree.ElementTree
import cv2 as cv
from geojson import Polygon, FeatureCollection, Feature
import numpy as np
import geopandas as gpd


def pretize_text(annotation_type):
    if annotation_type == 'blood_vessels':
        return 'Blood vessels'
    elif annotation_type == 'fatty_tissues':
        return 'Fatty tissue'
    elif annotation_type == 'inflammations':
        return 'Inflammation'
    elif annotation_type == 'endocardiums':
        return 'Endocardium'
    elif annotation_type == 'fibrotic_tissues':
        return 'Fibrotic tissue'
    elif annotation_type == 'quilities':
        return 'Quilty'
    elif annotation_type == 'immune_cells':
        return 'Immune cells'
    else:
        annotation_type = annotation_type.replace('_', ' ')
        return annotation_type.replace(annotation_type[0], annotation_type[0].upper(), 1)


def get_color(name):
    if name == 'blood_vessels':
        return [
            128,
            179,
            179
        ]
    elif name == 'endocardiums':
        return [
            240,
            154,
            16
        ]
    elif name == 'inflammations':
        return [
            255,
            255,
            153
        ]


def get_coors(contour):
    coors = []
    for idx in range(len(contour)):
        coors.append(contour[idx, 0].tolist())

    return coors


def fix_polygon(contour):
    return np.concatenate((contour, [contour[0]]))


def create_properties_template(annotation):
    return {
        "object_type": "annotation",
        "classification": {
            "name": pretize_text(annotation),
            "color": get_color(annotation),
        },
    }


def get_features(contours, annotation):
    features = []
    for contour in contours:
        contour = fix_polygon(contour)
        coors = get_coors(contour)
        if len(coors) <= 2:
            continue

        features.append(Feature(
            geometry=Polygon([coors]),
            properties=create_properties_template(annotation)
        ))

    return features


def create_geojson(mask, annotation_classes=None):
    if annotation_classes is None:
        annotation_classes = [
            'blood_vessels',
            'endocardiums',
            'fatty_tissues',
            'fibrotic_tissues',
            'immune_cells',
            'inflammations',
            'quilties'
        ]

    mask = np.uint8(mask * 255)

    features = []
    if len(mask.shape) == 3:
        _, _, classes = mask.shape
        assert classes == len(annotation_classes)

        for c in range(classes):
            contours, _ = cv.findContours(
                mask[:, :, c], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

            features.extend(get_features(contours, annotation_classes[c]))

        return FeatureCollection(features)
    else:
        assert len(annotation_classes) == 1

        contours, _ = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        features = get_features(contours, annotation_classes[0])

        return FeatureCollection(features)


def xml2geojson(xml_path):
    annotations = xml.etree.ElementTree.parse(xml_path)
    features = []

    for annotation in annotations.iter("Annotation"):
        g = annotation.get("PartOfGroup")
        coors = []
        for e in annotation.iter("Coordinate"):
            xy = [int(e.get("Y")), int(e.get("X"))]
            if sum(xy):
                coors.append(xy)

        features.append(Feature(
            geometry=Polygon([coors]),
            properties=create_properties_template(g)
        ))

    return FeatureCollection(features)


def postprocess_endocard(gdf, gdf_tissue):
    for idx, row in gdf.iterrows():
        if row['classification']['name'] == 'Endocardium' and gdf_tissue.boundary.distance(row.geometry.boundary).min() > 100:
            gdf.drop(idx, inplace=True)

    return gdf


def postprocess_inflammation(gdf, gdf_immune):
    for idx, row in gdf.iterrows():
        if row['classification']['name'] == 'Inflammation' and gdf_immune.within(row.geometry).sum() < 10:
            gdf.drop(idx, inplace=True)

    return gdf


def postprocess_vessels(gdf, gdf_tissue):
    for idx, row in gdf.iterrows():
        if row['classification']['name'] == 'Blood vessels' and any(gdf_tissue.boundary.intersects(row.geometry.boundary)):
            gdf.drop(idx, inplace=True)

    return gdf


def get_cell_mask(gj, shape):
    x, y = int(shape[0]), int(shape[1])

    mask = np.zeros((x, y), dtype='uint8')

    for feat in gj['features']:
        if feat['properties'].get('classification', None) is None or feat['properties']['classification']['name'] != 'Region*':
            geometry_name = 'nucleusGeometry' if feat.get(
                'nucleusGeometry') else 'geometry'
            coors = feat[geometry_name]['coordinates'][0]
            pts = [[round(c[0]), round(c[1])] for c in coors]
            cv.fillPoly(
                mask,
                [np.array(pts)],
                1
            )
    return mask


def get_cells(gj):
    immune_cells = []
    tissues = []

    for feat in gj['features']:
        if feat['properties'].get('classification', None) and feat['properties']['classification']['name'] == 'Region*':
            coors = feat['geometry']['coordinates']
            tissues.append(Feature(
                geometry=Polygon(coors)
            ))
        elif feat['properties'].get('classification', None) is None or feat['properties']['classification']['name'] != 'Region*':
            geometry_name = 'nucleusGeometry' if feat.get(
                'nucleusGeometry') else 'geometry'
            coors = feat[geometry_name]['coordinates']

            if feat['properties']['classification']['name'] == 'Immune cells':
                immune_cells.append(Feature(
                    geometry=Polygon(coors)
                ))

    gdf_tissue = gpd.GeoDataFrame.from_features(
        FeatureCollection(tissues))
    gdf_imunne = gpd.GeoDataFrame.from_features(
        FeatureCollection(immune_cells))
    return gdf_tissue, gdf_imunne


def get_mask_index(feature, classes):
    class_type = feature['classification']['name']

    for idx, name in enumerate(classes):
        if class_type.lower() == name.lower():
            return idx

    # else return Other cells
    return 0


def get_mask(shape, annotations, classes):
    x, y = int(shape[0]), int(shape[1])

    classes_masks = [
        np.zeros((x, y, 1), dtype='uint8')
        for _ in range(len(classes))
    ]

    for feat in annotations:
        geometry_name = 'geometry'
        coors = list(feat[geometry_name].exterior.coords)
        pts = [[round(c[0]), round(c[1])] for c in coors]
        cv.fillPoly(
            classes_masks[get_mask_index(feat, classes)],
            [np.array(pts)],
            1
        )

    mask = np.concatenate(classes_masks, axis=2)
    return mask
