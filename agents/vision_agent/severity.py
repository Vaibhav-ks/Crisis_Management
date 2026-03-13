def add_severity(zone_map,
                 flood_weight=0.6,
                 damage_weight=0.4):
    """
    Add severity score to each zone
    """

    for zone_id, data in zone_map.items():

        flood = data.get("flood_score", 0)
        damage = data.get("damage_score", 0)

        severity = (
            flood_weight * flood +
            damage_weight * damage
        )

        data["severity"] = round(severity, 3)

    return zone_map