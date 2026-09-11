import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json


@dataclass
class Survivor:
    id: int
    position: Tuple[float, float]
    condition: str  # 'trapped', 'injured', 'mobile', 'sheltered'
    detectability: float  # 0-1, 可检测性
    thermal_signal: float  # 热信号强度
    movement_signal: float  # 运动信号
    last_seen: int  # 最后被观察到的时间步
    accessibility: float  # 可达性 0-1


@dataclass
class Building:
    id: int
    position: Tuple[float, float]
    dimensions: Tuple[float, float]
    damage_level: float  # 0-1, 损坏程度
    building_type: str  # 'residential', 'commercial', 'industrial'
    structural_integrity: float  # 结构完整性


@dataclass
class Hazard:
    id: int
    position: Tuple[float, float]
    hazard_type: str  # 'fire', 'gas_leak', 'structural_instability'
    intensity: float  # 强度 0-1
    radius: float  # 影响半径


@dataclass
class Region:
    id: int
    boundary: List[Tuple[float, float]]  # 多边形边界
    region_type: str  # 'simple', 'complex', 'moderate'
    priority: float  # 优先级 0-1
    accessibility: float  # 可达性 0-1
    # 核心修改：删除 estimated_survivors，新增 victim_probability（受害者存在可能性）
    victim_probability: float  # 0-1, 区域内存在受害者的可能性


class SearchRescueSceneGenerator:
    def __init__(self, width=1000, height=1000):
        self.width = width
        self.height = height
        self.scene_id = f"scene_{random.randint(1000, 9999)}"

    def generate_complete_scene(self):
        """生成完整搜救场景"""
        print(f"生成搜救场景: {self.scene_id}")

        # 1. 生成地形分区（已含 victim_probability）
        regions = self._generate_terrain_regions()

        # 2. 生成建筑物
        buildings = self._generate_buildings(regions)

        # 3. 应用灾害损坏
        buildings = self._apply_disaster_damage(buildings, regions)

        # 4. 生成幸存者：基于区域 victim_probability 动态生成数量
        survivors = self._place_survivors(regions, buildings)

        # 5. 生成危险源
        hazards = self._generate_hazards(regions, buildings)

        # 6. 生成传感器信号
        sensor_signals = self._generate_sensor_signals(survivors, buildings, hazards)

        # 7. 创建场景对象
        scene = {
            'scene_id': self.scene_id,
            'dimensions': (self.width, self.height),
            'regions': regions,
            'buildings': buildings,
            'survivors': survivors,
            'hazards': hazards,
            'sensor_signals': sensor_signals,
            'timestamp': 0,  # 灾害发生后时间
            'weather_conditions': random.choice(['clear', 'cloudy', 'light_rain'])
        }

        return scene

    def _generate_terrain_regions(self):
        """生成地形分区：用 victim_probability 替代固定幸存者数量"""
        regions = []

        # 简单区域 - 开阔地带（受害者存在可能性低）
        simple_region = Region(
            id=1,
            boundary=[(0, 0), (400, 0), (400, 600), (0, 600)],
            region_type='simple',
            priority=0.2,
            accessibility=0.9,
            victim_probability=0.2  # 低可能性：开阔地带人员滞留少
        )
        regions.append(simple_region)

        # 中等复杂区域 - 轻度损坏区域（受害者存在可能性中等）
        moderate_region = Region(
            id=2,
            boundary=[(400, 0), (700, 0), (700, 500), (400, 500)],
            region_type='moderate',
            priority=0.5,
            accessibility=0.6,
            victim_probability=0.5  # 中等可能性：部分建筑损坏，可能有滞留人员
        )
        regions.append(moderate_region)

        # 复杂区域 - 严重损坏区域（受害者存在可能性高）
        complex_region = Region(
            id=3,
            boundary=[(700, 0), (1000, 0), (1000, 700), (400, 500), (700, 500)],
            region_type='complex',
            priority=0.9,
            accessibility=0.3,
            victim_probability=0.9  # 高可能性：建筑密集且严重损坏，易有被困者
        )
        regions.append(complex_region)

        return regions

    def _generate_buildings(self, regions):
        """在不同区域生成建筑物（逻辑不变）"""
        buildings = []
        building_id = 1

        # 简单区域 - 少量低层建筑
        simple_region = regions[0]
        for i in range(5):
            pos_x = random.randint(50, 350)
            pos_y = random.randint(50, 550)
            building = Building(
                id=building_id,
                position=(pos_x, pos_y),
                dimensions=(random.randint(30, 60), random.randint(30, 60)),
                damage_level=random.uniform(0.1, 0.3),
                building_type='residential',
                structural_integrity=random.uniform(0.7, 0.9)
            )
            buildings.append(building)
            building_id += 1

        # 中等区域 - 混合建筑
        moderate_region = regions[1]
        for i in range(8):
            pos_x = random.randint(420, 680)
            pos_y = random.randint(30, 470)
            building = Building(
                id=building_id,
                position=(pos_x, pos_y),
                dimensions=(random.randint(40, 80), random.randint(40, 80)),
                damage_level=random.uniform(0.3, 0.6),
                building_type=random.choice(['residential', 'commercial']),
                structural_integrity=random.uniform(0.4, 0.7)
            )
            buildings.append(building)
            building_id += 1

        # 复杂区域 - 密集高层建筑，严重损坏
        complex_region = regions[2]
        for i in range(12):
            pos_x = random.randint(720, 980)
            pos_y = random.randint(30, 670)
            building = Building(
                id=building_id,
                position=(pos_x, pos_y),
                dimensions=(random.randint(50, 100), random.randint(50, 100)),
                damage_level=random.uniform(0.6, 0.9),
                building_type=random.choice(['residential', 'commercial', 'industrial']),
                structural_integrity=random.uniform(0.1, 0.4)
            )
            buildings.append(building)
            building_id += 1

        return buildings

    def _apply_disaster_damage(self, buildings, regions):
        """应用灾害损坏模式（逻辑不变）"""
        for building in buildings:
            region_type = self._get_building_region(building, regions)
            if region_type == 'complex':
                building.damage_level *= 1.3  # 复杂区域损坏更严重
                building.structural_integrity *= 0.7
            elif region_type == 'moderate':
                building.damage_level *= 1.1

        return buildings

    def _place_survivors(self, regions, buildings):
        """核心修改：基于区域 victim_probability 动态生成幸存者数量"""
        survivors = []
        survivor_id = 1

        for region in regions:
            # 根据 victim_probability 确定该区域幸存者数量范围
            # 逻辑：可能性越高，生成的幸存者数量越多（含随机波动）
            if region.victim_probability < 0.3:  # 低可能性（如简单区域）
                min_survivors = 1
                max_survivors = 3
            elif 0.3 <= region.victim_probability < 0.7:  # 中等可能性（如中等区域）
                min_survivors = 3
                max_survivors = 6
            else:  # 高可能性（如复杂区域）
                min_survivors = 8
                max_survivors = 12

            # 随机生成该区域的实际幸存者数量
            region_survivor_count = random.randint(min_survivors, max_survivors)

            # 生成该区域的幸存者
            for _ in range(region_survivor_count):
                # 部分在建筑内，部分在室外（逻辑不变，适配区域边界）
                if random.random() < 0.6:
                    # 在该区域的建筑内
                    region_buildings = [b for b in buildings
                                       if self._get_building_region(b, regions) == region.region_type]
                    if region_buildings:  # 确保区域有建筑
                        building = random.choice(region_buildings)
                        pos_x = building.position[0] + random.uniform(-20, 20)
                        pos_y = building.position[1] + random.uniform(-20, 20)
                    else:  # 无建筑则在室外
                        pos_x = self._get_random_point_in_region(region)[0]
                        pos_y = self._get_random_point_in_region(region)[1]
                else:
                    # 在该区域的室外
                    pos_x = self._get_random_point_in_region(region)[0]
                    pos_y = self._get_random_point_in_region(region)[1]

                # 根据区域类型调整幸存者状态（复杂区域更多被困者，逻辑不变）
                if region.region_type == 'simple':
                    condition = random.choice(['mobile', 'sheltered'])
                    detectability = random.uniform(0.8, 1.0)
                    thermal_signal = random.uniform(0.7, 0.9)
                    movement_signal = random.uniform(0.6, 0.9)
                    accessibility = random.uniform(0.8, 1.0)
                elif region.region_type == 'moderate':
                    condition = random.choice(['injured', 'sheltered', 'mobile'])
                    detectability = random.uniform(0.5, 0.8)
                    thermal_signal = random.uniform(0.4, 0.7)
                    movement_signal = random.uniform(0.3, 0.6)
                    accessibility = random.uniform(0.5, 0.8)
                else:  # complex
                    condition = random.choice(['trapped', 'injured', 'trapped'])
                    detectability = random.uniform(0.1, 0.5)
                    thermal_signal = random.uniform(0.1, 0.4)
                    movement_signal = random.uniform(0.1, 0.3)
                    accessibility = random.uniform(0.1, 0.4)

                survivor = Survivor(
                    id=survivor_id,
                    position=(pos_x, pos_y),
                    condition=condition,
                    detectability=detectability,
                    thermal_signal=thermal_signal,
                    movement_signal=movement_signal,
                    last_seen=random.randint(-10, 0) if region.region_type != 'simple' else 0,
                    accessibility=accessibility
                )
                survivors.append(survivor)
                survivor_id += 1

        return survivors

    def _get_random_point_in_region(self, region):
        """辅助函数：在区域边界内生成随机点（确保室外幸存者在区域内）"""
        # 简化逻辑：取区域边界的x、y范围，生成随机点（适用于矩形/凸多边形区域）
        x_coords = [p[0] for p in region.boundary]
        y_coords = [p[1] for p in region.boundary]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        # 循环确保点在多边形内（避免边界不规则导致点超出）
        while True:
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            if self._point_in_polygon(x, y, region.boundary):
                return (x, y)

    def _generate_hazards(self, regions, buildings):
        """生成危险源（逻辑不变）"""
        hazards = []
        hazard_id = 1

        # 在复杂区域生成更多危险
        complex_buildings = [b for b in buildings if b.damage_level > 0.7]
        for building in complex_buildings[:4]:
            # 火灾危险
            if random.random() < 0.7:
                hazard = Hazard(
                    id=hazard_id,
                    position=building.position,
                    hazard_type='fire',
                    intensity=random.uniform(0.6, 0.9),
                    radius=random.uniform(40, 80)
                )
                hazards.append(hazard)
                hazard_id += 1

            # 结构危险
            if random.random() < 0.8:
                hazard = Hazard(
                    id=hazard_id,
                    position=building.position,
                    hazard_type='structural_instability',
                    intensity=random.uniform(0.7, 1.0),
                    radius=random.uniform(30, 60)
                )
                hazards.append(hazard)
                hazard_id += 1

        # 在中等区域生成一些危险
        moderate_buildings = [b for b in buildings if 0.4 < b.damage_level <= 0.7]
        for building in moderate_buildings[:2]:
            if random.random() < 0.5:
                hazard = Hazard(
                    id=hazard_id,
                    position=building.position,
                    hazard_type='gas_leak',
                    intensity=random.uniform(0.4, 0.7),
                    radius=random.uniform(50, 100)
                )
                hazards.append(hazard)
                hazard_id += 1

        return hazards

    def _generate_sensor_signals(self, survivors, buildings, hazards):
        """生成传感器信号（逻辑不变）"""
        sensor_signals = {
            'thermal_signatures': [],
            'movement_detections': [],
            'acoustic_signals': [],
            'visual_detections': []
        }

        # 生成热信号
        for survivor in survivors:
            if survivor.thermal_signal > 0.3:
                sensor_signals['thermal_signatures'].append({
                    'position': survivor.position,
                    'intensity': survivor.thermal_signal,
                    'certainty': min(1.0, survivor.thermal_signal * 1.2),
                    'source_type': 'survivor'
                })

        # 生成运动检测
        for survivor in survivors:
            if survivor.movement_signal > 0.2:
                sensor_signals['movement_detections'].append({
                    'position': survivor.position,
                    'intensity': survivor.movement_signal,
                    'certainty': survivor.movement_signal * 0.8,
                    'timestamp': random.randint(-10, 0)
                })

        # 生成火灾的热信号
        for hazard in hazards:
            if hazard.hazard_type == 'fire':
                sensor_signals['thermal_signatures'].append({
                    'position': hazard.position,
                    'intensity': hazard.intensity,
                    'certainty': 0.95,
                    'source_type': 'fire'
                })

        # 生成声学信号
        for survivor in survivors:
            if survivor.condition in ['injured', 'trapped'] and random.random() < 0.4:
                sensor_signals['acoustic_signals'].append({
                    'position': survivor.position,
                    'intensity': random.uniform(0.3, 0.7),
                    'certainty': random.uniform(0.4, 0.8),
                    'type': 'human_voice'
                })

        return sensor_signals

    def _get_building_region(self, building, regions):
        """确定建筑物所在区域类型（逻辑不变）"""
        x, y = building.position
        for region in regions:
            if self._point_in_polygon(x, y, region.boundary):
                return region.region_type
        return 'unknown'

    def _point_in_polygon(self, x, y, poly):
        """判断点是否在多边形内（逻辑不变）"""
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside


# 可视化场景（仅基础场景，逻辑不变）
def visualize_scene(scene):
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    ax1.set_title(f"搜救场景 - {scene['scene_id']}")
    ax1.set_xlim(0, scene['dimensions'][0])
    ax1.set_ylim(0, scene['dimensions'][1])

    # 绘制区域（新增显示 victim_probability）
    colors = {'simple': 'lightgreen', 'moderate': 'gold', 'complex': 'lightcoral'}
    for region in scene['regions']:
        polygon = plt.Polygon(region.boundary, alpha=0.3,
                              color=colors[region.region_type],
                              label=region.region_type)
        ax1.add_patch(polygon)

        # 区域标签：显示受害者存在可能性（替换原估计幸存者数量）
        centroid = np.mean(region.boundary, axis=0)
        ax1.text(centroid[0], centroid[1],
                 f"{region.region_type}\n优先级: {region.priority:.2f}\n存在可能性: {region.victim_probability:.2f}",
                 ha='center', va='center', fontweight='bold', fontsize=9)

    # 绘制建筑物
    for building in scene['buildings']:
        color_intensity = np.clip(building.damage_level, 0.0, 1.0)
        color = (color_intensity, 0, 0)
        rect = Rectangle(building.position,
                         building.dimensions[0], building.dimensions[1],
                         facecolor=color, alpha=0.7, edgecolor='black')
        ax1.add_patch(rect)

    # 绘制幸存者
    condition_colors = {
        'trapped': 'red', 'injured': 'orange',
        'mobile': 'blue', 'sheltered': 'green'
    }
    for survivor in scene['survivors']:
        color = condition_colors[survivor.condition]
        circle = Circle(survivor.position, radius=8,
                        facecolor=color, alpha=0.8, edgecolor='black')
        ax1.add_patch(circle)

        # 检测性指示圈
        detect_circle = Circle(survivor.position,
                               radius=12 * survivor.detectability,
                               facecolor='none',
                               edgecolor=color,
                               linestyle='--',
                               alpha=0.5)
        ax1.add_patch(detect_circle)

    # 绘制危险源
    hazard_colors = {'fire': 'red', 'gas_leak': 'yellow', 'structural_instability': 'brown'}
    for hazard in scene['hazards']:
        circle = Circle(hazard.position, radius=hazard.radius,
                        facecolor='none',
                        edgecolor=hazard_colors[hazard.hazard_type],
                        linewidth=2, linestyle='-',
                        alpha=0.7)
        ax1.add_patch(circle)
        ax1.text(hazard.position[0], hazard.position[1],
                 hazard.hazard_type[:3], ha='center', va='center',
                 fontweight='bold')

    ax1.legend()
    ax1.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def print_scene_summary(scene):
    """打印场景摘要：显示 victim_probability，删除原估计幸存者数量"""
    print("\n" + "=" * 50)
    print(f"场景摘要: {scene['scene_id']}")
    print("=" * 50)

    # 区域统计：显示受害者存在可能性 + 实际幸存者数量
    print("\n区域统计:")
    for region in scene['regions']:
        # 统计该区域实际幸存者数量
        survivors_in_region = len([s for s in scene['survivors']
                                   if generator._point_in_polygon(s.position[0], s.position[1], region.boundary)])
        print(f"  {region.region_type:>8}区域: 优先级{region.priority:.2f}, "
              f"可达性{region.accessibility:.2f}, 受害者存在可能性{region.victim_probability:.2f}, "
              f"实际幸存者{survivors_in_region}人")

    # 幸存者统计（逻辑不变）
    print(f"\n幸存者统计 (总计: {len(scene['survivors'])}人):")
    conditions = {}
    for survivor in scene['survivors']:
        conditions[survivor.condition] = conditions.get(survivor.condition, 0) + 1

    for condition, count in conditions.items():
        print(f"  {condition}: {count}人")

    # 搜救任务建议：基于 victim_probability 优先（原优先级逻辑不变，更贴合可能性）
    print(f"\n搜救任务建议:")
    # 按“受害者存在可能性+优先级”排序，优先高可能性高优先级区域
    priority_regions = sorted(scene['regions'],
                             key=lambda r: (r.victim_probability, r.priority),
                             reverse=True)
    for region in priority_regions[:2]:  # 推荐前2个高优先级区域
        survivors_count = len([s for s in scene['survivors']
                               if generator._point_in_polygon(s.position[0], s.position[1], region.boundary)])
        print(f"  ⚠️ 优先搜救{region.region_type}区域: 受害者存在可能性{region.victim_probability:.2f}, "
              f"估计{survivors_count}名幸存者")


# 生成并显示场景
if __name__ == "__main__":
    generator = SearchRescueSceneGenerator()
    scene = generator.generate_complete_scene()

    # 可视化场景
    visualize_scene(scene)

    # 打印详细摘要
    print_scene_summary(scene)

    # 保存场景数据：适配 victim_probability 字段
    with open(f"{scene['scene_id']}.json", 'w') as f:
        serializable_scene = {
            'scene_id': scene['scene_id'],
            'dimensions': scene['dimensions'],
            'regions': [{
                'id': r.id,
                'boundary': r.boundary,
                'region_type': r.region_type,
                'priority': r.priority,
                'accessibility': r.accessibility,
                'victim_probability': r.victim_probability  # 替换原 estimated_survivors
            } for r in scene['regions']],
            'survivors': [{
                'id': s.id,
                'position': s.position,
                'condition': s.condition,
                'detectability': s.detectability,
                'thermal_signal': s.thermal_signal,
                'movement_signal': s.movement_signal,
                'accessibility': s.accessibility
            } for s in scene['survivors']],
            'hazards': [{
                'id': h.id,
                'position': h.position,
                'hazard_type': h.hazard_type,
                'intensity': h.intensity,
                'radius': h.radius
            } for h in scene['hazards']]
        }
        json.dump(serializable_scene, f, indent=2)

    print(f"\n场景数据已保存到: {scene['scene_id']}.json")