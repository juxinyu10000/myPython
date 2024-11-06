import pygame
import random

# 初始化pygame
pygame.init()

# 设置窗口
WIDTH, HEIGHT = 480, 640
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("飞机大战")

# 加载图片
background = pygame.image.load("./test_image/background.jpg")
player_img = pygame.image.load("./test_image/player.png")
enemy_img = pygame.image.load("./test_image/enemy.png")
bullet_img = pygame.image.load("./test_image/bullet.png")

# 设置颜色
WHITE = (255, 255, 255)


# 玩家类
class Player:
    def __init__(self):
        self.image = player_img
        # 初始化矩形对象，用于定义图像的位置和尺寸
        self.rect = self.image.get_rect()
       # 设置矩形的中心位置，位于窗口宽度的中心和高度减去50的地方。
        self.rect.center = (WIDTH // 2, HEIGHT - 50)


    def move(self, dx):
        self.rect.x += dx
        if self.rect.x < 0:
            self.rect.x = 0
        if self.rect.x > WIDTH - self.rect.width:
            self.rect.x = WIDTH - self.rect.width

    def draw(self, surface):
        surface.blit(self.image, self.rect)


# 敌机类
class Enemy:
    def __init__(self):
        self.image = enemy_img
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, WIDTH - self.rect.width)
        self.rect.y = random.randint(-100, -40)

    def move(self):
        self.rect.y += 2
        if self.rect.y > HEIGHT:
            self.rect.y = random.randint(-100, -40)
            self.rect.x = random.randint(0, WIDTH - self.rect.width)

    def draw(self, surface):
        surface.blit(self.image, self.rect)


# 子弹类
class Bullet:
    def __init__(self, x, y):
        self.image = bullet_img
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

    def move(self):
        self.rect.y -= 4

    def draw(self, surface):
        surface.blit(self.image, self.rect)





# 主程序
def main():
    enemies_num = 3
    clock = pygame.time.Clock()
    player = Player()
    enemies = [Enemy() for _ in range(enemies_num)]
    bullets = []
    running = True

    # # 新增变量，以确保每次按空格键只发射一枚子弹
    # bullet_fired = False

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            player.move(-4)
        if keys[pygame.K_RIGHT]:
            player.move(4)
        if keys[pygame.K_SPACE]: # and not bullet_fired:  # 检查是否发射子弹
            bullets.append(Bullet(player.rect.centerx, player.rect.top))
            # bullet_fired = True  # 设置为 True，标记已发射子弹

        # 更新子弹位置
        for bullet in bullets[:]:
            bullet.move()
            if bullet.rect.y < 0:
                bullets.remove(bullet)
                bullet_fired = False  # 当子弹出屏时，允许再次发射

        # 更新敌机位置
        for enemy in enemies[:]:
            enemy.move()

            # 碰撞检测
            for bullet in bullets[:]:
                if bullet.rect.colliderect(enemy.rect):
                    bullets.remove(bullet)  # 移除子弹
                    enemies.remove(enemy)  # 移除敌机
                    bullet_fired = False  # 碰撞后也重置发射状态
                    break  # 一旦找到碰撞，跳出循环，避免修改列表时的错误

        if len(enemies) == 0:
            enemies_num += 1
            if enemies_num == 10:
                enemies_num = 3
            enemies = [Enemy() for _ in range(enemies_num)]

        # 绘制
        screen.blit(background, (0, 0))
        player.draw(screen)
        for enemy in enemies:
            enemy.draw(screen)
        for bullet in bullets:
            bullet.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()