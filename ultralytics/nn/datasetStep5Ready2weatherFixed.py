
# ================= 配置区域 =================


# 雨型预设配置
RAIN_PROFILES = {
    "暴雨": {
        # --- 几何参数 ---
        "drop_length": (50, 140),   # 雨滴长度范围。越大雨丝越长，视觉噪声越明显
        "width_range": (1.5, 2.9),   # 雨滴宽度范围。越大雨丝越粗，遮挡感越强
        "base_angle": 83,            # 基础角度（度）。接近90为垂直，越小倾斜越厉害
        "angle_variation": 5,        # 角度随机波动范围。越大雨丝方向越杂乱
        
        # --- 密度与分布 ---
        "layers": 3,                 # 叠加层数。越大雨越密集，噪声越多
        "grid_size": 20,             # 网格大小。越小雨滴分布越密集（计算量增加）
        "density_factor": 0.15,      # 密度系数 (0-1)。越大留存的雨滴越多，噪声越强
        "offset_variation": 10,      # 网格内随机偏移量。越大分布越自然，避免过于整齐
        
        # --- 外观参数 ---
        "alpha": 140,                 # 透明度 (0-255)。越大雨滴越白/显眼，噪声对比度越高
        "blur_radius": 2,            # 单雨滴模糊半径。越大边缘越柔和，但仅限雨滴局部，不影响背景
    }
}

# 雾效预设配置
FOG_CONFIG = {
    # 【修改】颜色改为随机灰度范围，值越大雾色越亮
    "color_range": (150, 225),       # 雾颜色随机灰度值范围 (0-255)
    # 【修改】扩大噪声强度，让雾气分布更不均匀/斑驳
    "noise_range": (20, 60),         # 深度图噪声强度。越大雾气分布越不均匀/斑驳
    # 注意：雾效会改变全图像素，因此没有"mask"概念，但不再进行全图模糊
}

# 雪效预设配置
SNOW_CONFIG = {
    "num_circles": 200,              # 圆形雪花数量。越大雪越密
    "num_ellipses": 2000,            # 椭圆雪花数量。越大动态模糊感越强
    "frame_count": 1,                # 叠加帧数。越大雪花轨迹越丰富，噪声越多
    "color_range": (220, 255),       # 雪花灰度值范围 (0-255)，越接近255越白
}


# ================= 类定义 =================

class SnowParticle:
    def __init__(self, image_size):
        self.w, self.h = image_size
        # 【修改1】雪花面积扩大4倍 -> 线性尺寸扩大2倍
        self.max_width = 20     # 原 6 * 2
        self.min_width = 5      # 原 1 * 2
        self.max_speed = 2     # 最大速度
        self.min_speed = 1     # 最低速度
        self.reset_properties()
    
    def reset_properties(self):
        """初始化粒子属性"""
        if random.random() < 0.8: 
            # 80% 的雪花是小尺寸
            self.width = random.randint(self.min_width, 8)
        else:
            # 20% 的雪花是大尺寸
            self.width = random.randint(8, self.max_width)
        
        # 【新增】为雪花设置随机灰度和透明度
        min_c, max_c = SNOW_CONFIG["color_range"]
        self.color = random.randint(min_c, max_c)
        self.opacity = random.uniform(0.4, 0.9) # 透明度在 0.4 到 0.9 之间随机

        self.x = random.uniform(-self.w*0.2, self.w*1.2)
        self.y = random.uniform(-self.h*0.5, self.h*1.5)
        self.speed = random.uniform(self.min_speed, self.max_speed)
        # 动画参数
        self.angle = math.radians(random.uniform(-15, 15))
        self.swing_freq = random.uniform(0.02, 0.05)

    def create_solid_snowflake(self):
        """创建实心圆形雪花"""
        size = int(self.width * 1.5)
        solid = Image.new('RGBA', (size, size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(solid)
        
        alpha = int(255 * self.opacity)
        draw.ellipse(
            [(0, 0), (size, size)],
            fill=(255, 255, 255, alpha),
            outline=(255, 255, 255, alpha)
        )
        return solid.resize((self.width, self.width), Image.LANCZOS)

    def create_ellipse_snowflake(self):
        """创建椭圆形雪花，模拟快速下落的样子"""
        size = int(self.width * 1.5)
        ellipse = Image.new('RGBA', (size, size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(ellipse)
        
        alpha = int(255 * self.opacity)
        c = self.color
        stretch = max(1, self.speed * 1.5)
        width = size // 2
        height = int(size // 2 * stretch)
        
        bbox = (
            size // 2 - width // 2,
            size // 2 - height // 2,
            size // 2 + width // 2,
            size // 2 + height // 2
        )
        draw.ellipse(bbox, fill=(c, c, c, alpha), outline=(c, c, c, alpha))
        
        return ellipse.resize((self.width, self.width), Image.LANCZOS)

    def calculate_position(self, frame):
        """计算运动轨迹"""
        self.y += self.speed * 2
        self.x += (math.sin(frame * self.swing_freq) * 10 +
                  math.cos(self.angle) * self.speed * 0.5)
        
        # 边界重置
        if self.y > self.h * 1.2:
            self.reset_properties()


# ================= 核心生成逻辑 =================

def generate_blizzard_frame_strict(image_size, frame_num):
    """
    生成单帧雪花效果（严格模式：无全图模糊）
    """
    frame = Image.new('RGBA', image_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(frame)
    
    num_circles = SNOW_CONFIG["num_circles"]
    num_ellipses = SNOW_CONFIG["num_ellipses"]
    
    # 创建圆形雪花组
    circle_particles = [SnowParticle(image_size) for _ in range(num_circles)]
    for p in circle_particles:
        p.type = 0 
    
    # 创建椭圆雪花组
    ellipse_particles = [SnowParticle(image_size) for _ in range(num_ellipses)]
    for p in ellipse_particles:
        p.type = 1 
    
    particles = circle_particles + ellipse_particles
    
    for p in particles:
        for _ in range(2):  # 模拟2帧运动轨迹
            p.calculate_position(frame_num)
            if p.type == 0:
                snowflake = p.create_solid_snowflake()
            else:
                snowflake = p.create_ellipse_snowflake()
            
            # 直接 paste，利用 snowflake 自身的 alpha 通道
            # 注意：这里不对整张 frame 做 GaussianBlur，以保护背景像素
            frame.paste(snowflake, (int(p.x), int(p.y)), snowflake)
    
    return frame

def create_rain_layer_strict(size, profile):
    """
    创建更整齐的雨层（严格模式：返回带Alpha通道的图层，不做全图处理）
    """
    layer = Image.new('RGBA', size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(layer)
    
    base_angle = profile["base_angle"]
    grid_size = profile["grid_size"]
    width, height = size

    for gy in range(-2, height // grid_size + 2):
        for gx in range(-2, width // grid_size + 2):
            if random.random() > profile["density_factor"]:
                continue
                
            offset = profile["offset_variation"]
            x = gx * grid_size + random.randint(-offset, offset)
            y = gy * grid_size + random.randint(-offset, offset)
            
            if x < 0 or x >= width or y < 0 or y >= height:
                continue

            length = random.uniform(*profile["drop_length"])
            width_val = random.uniform(*profile["width_range"])
            
            angle = base_angle + random.randint(-profile["angle_variation"], profile["angle_variation"])
            rad = np.deg2rad(angle)
            dx, dy = np.cos(rad), np.sin(rad)
            
            end_x = x + dx * length
            end_y = y + dy * length
            
            draw.line([(x, y), (end_x, end_y)],
                     fill=(255, 255, 255, profile["alpha"]),
                     width=int(width_val))
    
    # 仅对当前雨滴层进行轻微模糊，使雨滴边缘柔和，但这不会影响背景
    # 因为模糊是在透明背景上进行的，扩散出的像素也是半透明的，后续会通过Mask严格控制
    if profile["blur_radius"] > 0:
        return layer.filter(ImageFilter.GaussianBlur(profile["blur_radius"]))
    return layer


# ================= 统一天气接口 =================

def add_weather_noise(image, weather_type='rain'):
    """
    为图像添加天气噪声的统一接口
    
    Args:
        image: PIL Image 对象 (RGB)
        weather_type: 'rain', 'snow', 或 'fog'
        
    Returns:
        PIL Image 对象 (RGB)
    """
    # 【修改2】固定随机种子，确保每次调用生成的噪声图案完全一致
    seedNum =    random.randint(0,10)
    print(seedNum)
    random.seed(seedNum)
    np.random.seed(seedNum)
    
    img_rgb = image.convert('RGB')
    width, height = img_rgb.size
    
    if weather_type == 'rain':
        rain_type = "暴雨" # 固定使用暴雨配置，配合固定种子保证一致性
        profile = RAIN_PROFILES[rain_type]
        
        # 1. 创建雨滴复合层 (RGBA)
        rain_composite = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        for i in range(profile["layers"]):
            layer_profile = profile.copy()
            layer_profile["base_angle"] = profile["base_angle"] + random.randint(-1, 1)
            rain_layer = create_rain_layer_strict((width, height), layer_profile)
            rain_composite = Image.alpha_composite(rain_composite, rain_layer)
        
        # 2. 严格 Mask 粘贴
        img_rgba = img_rgb.convert('RGBA')
        # 使用 rain_composite 的 Alpha 通道作为 Mask
        # 只有 Alpha > 0 的地方才会被修改，其他地方完全保留原图像素
        img_rgba.paste(rain_composite, (0, 0), rain_composite)
        return img_rgba.convert('RGB')

    elif weather_type == 'snow':
        # 1. 创建雪花复合层 (RGBA)
        snow_composite = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        for i in range(SNOW_CONFIG["frame_count"]):
            frame = generate_blizzard_frame_strict((width, height), i)
            snow_composite = Image.alpha_composite(snow_composite, frame)



        # 2. 严格 Mask 粘贴
        img_rgba = img_rgb.convert('RGBA')
        # 使用 snow_composite 的 Alpha 通道作为 Mask
        img_rgba.paste(snow_composite, (0, 0), snow_composite)
        return img_rgba.convert('RGB')

    elif weather_type == 'fog':
        # 【修改3】实现随机颜色和增强的分布不均匀性
        config = FOG_CONFIG
        
        # 从配置范围中随机选择一个灰度值
        min_val, max_val = config["color_range"]
        gray_value = random.randint(min_val, max_val)
        fog_color = (gray_value, gray_value, gray_value)
        
        noise_intensity = random.randint(*config["noise_range"])
        
        # 1. 创建具有随机颜色的雾层
        fog = Image.new('RGB', (width, height), fog_color)
        
        # 2. 生成高级深度图 (Mask)，模拟少数大片成团的雾气
        # 创建一个黑色背景用于绘制雾团
        patch_mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(patch_mask)

        # 定义要生成的雾团数量
        num_patches = random.randint(2, 3)

        for _ in range(num_patches):
            # 随机确定雾团的中心、大小和核心亮度
            patch_cx = random.randint(0, width)
            patch_cy = random.randint(0, height)
            patch_radius = random.randint(width // 4, width // 2)
            patch_brightness = random.randint(180, 255)
            
            # 绘制一个椭圆作为雾团的基础形状
            bbox = [patch_cx - patch_radius, patch_cy - patch_radius, 
                    patch_cx + patch_radius, patch_cy + patch_radius]
            draw.ellipse(bbox, fill=patch_brightness)

        # 【关键】使用非常大的高斯模糊来使雾团边缘极其柔和，融合成片
        cloud_mask = patch_mask.filter(ImageFilter.GaussianBlur(radius=width // 15))

        # 创建从下到上的垂直梯度
        depth_gradient = np.linspace(255, 0, height).reshape(height, 1)
        depth_gradient = np.tile(depth_gradient, (1, width)).astype(np.uint8)
        
        # 将雾团和垂直梯度结合，让雾气既有团状感，又有层次感
        combined_mask_array = (np.array(cloud_mask) * 0.8 + depth_gradient * 0.2).astype(np.uint8)
        depth_mask = Image.fromarray(combined_mask_array, mode='L')
        
        # 3. 混合计算
        img_array = np.array(img_rgb, dtype=np.float32) / 255.0
        fog_array = np.array(fog, dtype=np.float32) / 255.0
        alpha = (np.array(depth_mask, dtype=np.float32) / 255.0)[..., np.newaxis]
        
        blended = img_array * (1 - alpha) + fog_array * alpha
        result_img = Image.fromarray((blended * 255).astype(np.uint8))
        
        return result_img

    else:
        raise ValueError(f"Unsupported weather type: {weather_type}")

