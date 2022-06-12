import pygame


# справа отображаются квадраты, символизирующие список сомнительных пикселей
# щелчок по такому квадрату показывает окрестность пикселя
# нажатие цифровых клавиш от 1..4 задает класс пикселя от 0..3
# указанные классы записываются в маску по координатам пикселя


def show_active_learning(bad_pixels, image, mask):
    # храним метки классов
    class_list = [0] * len(bad_pixels)

    bg_color = (0, 0, 0, 0)  # Цвет фона с альфа-каналом 0 - прозрачный
    (width, height) = (640, 480)

    pygame.display.set_caption("Active learning")
    screen = pygame.display.set_mode((width, height))

    imageSize = image.get_size()

    # Вычисляем коэффициенты масштабирования картинки до размеров окна
    scale = (imageSize[0] / width, imageSize[1] / height)

    # масштабируем картинку до размеров окна
    myImage = pygame.transform.scale(image, (width - 40, height))

    # создаем экран для рисования полупрозрачных точек-кружков, для этого добавляем альфа-канал
    surface = screen.convert_alpha()

    surface.fill(bg_color)

    screen.blit(myImage, (0, 0))

    _currentPoint = [0, 0]

    for count_pixel in range(0, 10):
        pygame.draw.rect(surface, (255, 255, 255), (600, count_pixel * 40, 40, 38))
        pygame.draw.rect(surface, (0, 0, 0), (600, count_pixel * 40, 40, 38), 2)

    pygame.font.init()
    my_font = pygame.font.SysFont('Comic Sans MS', 30)

    running = True

    screen.blit(surface, (0, 0))
    pygame.display.flip()

    def mouse_in_rect(pos):
        if width > pos[0] > width - 40:
            for number_rect in range(0, 10):
                if number_rect * 40 < pos[1] < (number_rect + 1) * 40:
                    return number_rect
        return -1

    def draw_bad_pixel(number_rect):
        global _currentPoint
        if number_rect == -1:
            return
        screen.blit(myImage, (0, 0))
        surface.fill((0, 0, 0, 0))
        _currentPoint = [bad_pixels[number_rect][0] / scale[0], bad_pixels[number_rect][1] / scale[1]]
        pygame.draw.circle(surface, (0, 200, 0, 100), _currentPoint, 8)
        screen.blit(surface, (0, 0))

    def draw_class_label(pos, class_number):
        pygame.draw.rect(screen, (255, 255, 255), (600, pos * 40, 40, 38))
        pygame.draw.rect(screen, (0, 0, 0), (600, pos * 40, 40, 38), 2)
        text = my_font.render(str(class_number), False, (0, 0, 0))
        screen.blit(text, (600 + 40 / 2, pos * 40 + 10))

    _pos = -1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                _pos = mouse_in_rect(pygame.mouse.get_pos())
                draw_bad_pixel(_pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    if _pos != -1:
                        draw_class_label(_pos, 1)
                    class_list[_pos] = 0
                    _pos = -1
                if event.key == pygame.K_2:
                    if _pos != -1:
                        draw_class_label(_pos, 2)
                    class_list[_pos] = 1
                    _pos = -1
                if event.key == pygame.K_3:
                    if _pos != -1:
                        draw_class_label(_pos, 3)
                    class_list[_pos] = 2
                    _pos = -1
                if event.key == pygame.K_4:
                    if _pos != -1:
                        draw_class_label(_pos, 4)
                    class_list[_pos] = 3
                    _pos = -1

        pygame.display.update()

    pygame.quit()

    for idx, bad_pixel in enumerate(bad_pixels):
        mask[bad_pixel[0], bad_pixel[1]] = class_list[idx]

    return mask


#############################################################################################################
# TEST
#############################################################################################################

# # test image
# _myImage = pygame.image.load('../passport_dataset/data/3828614.jpg')
# imageSize = _myImage.get_size()
#
# # test mask
# _mask = torch.zeros(imageSize[0], imageSize[1])
#
# _bad_pixels = []
#
# # test pixels
# for bad_pixels_count in range(0, 10):
#     _bad_pixels.append([random.randint(0, 250 - 1), random.randint(0, 250 - 1)])
#
# # result mask
# mask = show_active_learning(_bad_pixels, _myImage, _mask)
