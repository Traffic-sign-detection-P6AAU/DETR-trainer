import unittest
from data_handler.data_split import find_annos_and_imgs, divide_data

ACCEPTED_CATS = [
    {
        'id': 1,
        'oldid': 1
    },
    {
        'id': 2,
        'oldid': 2
    },
    {
        'id': 3,
        'oldid': 3
    }
]

class FindAnnosAndImgsTests(unittest.TestCase):
    def test_imgs_id_off(self):
        labels = {
            'annotations': [
                {'image_id': 1, 'category_id': 1},
                {'image_id': 2, 'category_id': 2}
            ],
            'categories': ACCEPTED_CATS,
            'images': [
                {'id': 1},
                {'id': 3}
            ]
        }
        cat_ids = [1, 2]
        expected = {
           'annotations': [
                {'image_id': 1, 'category_id': 1}
           ],
           'categories': ACCEPTED_CATS,
            'images': [
                {'id': 1}
            ]
        }
        result = find_annos_and_imgs(labels, cat_ids, ACCEPTED_CATS)
        self.assertEqual(result, expected)

    def test_image_id_off(self):
        labels = {
            'annotations': [
                {'image_id': 1, 'category_id': 1},
                {'image_id': 3, 'category_id': 2}
            ],
            'categories': ACCEPTED_CATS,
            'images': [
                {'id': 1},
                {'id': 2}
            ]
        }
        cat_ids = [1, 2]
        expected = {
           'annotations': [
                {'image_id': 1, 'category_id': 1}
           ],
           'categories': ACCEPTED_CATS,
            'images': [
                {'id': 1}
            ]
        }
        result = find_annos_and_imgs(labels, cat_ids, ACCEPTED_CATS)
        self.assertEqual(result, expected)

    def test_image_id_mis(self):
        labels = {
            'annotations': [
                {'image_id': 2, 'category_id': 2}
            ],
            'categories': ACCEPTED_CATS,
            'images': [
                {'id': 1},
                {'id': 2}
            ]
        }
        cat_ids = [1, 2]
        expected = {
           'annotations': [
                {'image_id': 2, 'category_id': 2}
           ],
           'categories': ACCEPTED_CATS,
            'images': [
                {'id': 2}
            ]
        }
        result = find_annos_and_imgs(labels, cat_ids, ACCEPTED_CATS)
        self.assertEqual(result, expected)
        
    def test_cat_id_off(self):
        labels = {
            'annotations': [
                {'image_id': 1, 'category_id': 1},
                {'image_id': 2, 'category_id': 3}
            ],
            'categories': ACCEPTED_CATS,
            'images': [
                {'id': 1},
                {'id': 2}
            ]
        }
        cat_ids = [1, 2]
        expected = {
           'annotations': [
                {'image_id': 1, 'category_id': 1}
           ],
            'categories': ACCEPTED_CATS,
            'images': [
                {'id': 1}
            ]
        }
        result = find_annos_and_imgs(labels, cat_ids, ACCEPTED_CATS)
        self.assertEqual(result, expected)

    def test_all_good(self):
        labels = {
            'annotations': [
                {'image_id': 1, 'category_id': 1},
                {'image_id': 5, 'category_id': 3}
            ],
            'categories': ACCEPTED_CATS,
            'images': [
                {'id': 1},
                {'id': 5}
            ]
        }
        cat_ids = [1, 3]
        result = find_annos_and_imgs(labels, cat_ids, ACCEPTED_CATS)
        self.assertEqual(result, labels)

class DivideDataTest(unittest.TestCase):
    def test_all_good(self):
        data = {
            'annotations': [
                {'image_id': 1, 'category_id': 1},
                {'image_id': 5, 'category_id': 3}
            ],
            'images': [
                {'id': 1},
                {'id': 5}
            ]
        }
        cat_ids = [1, 3]
        expected = (
            {
                'annotations': [
                    {'image_id': 1, 'category_id': 1}
                ],
                'categories': ACCEPTED_CATS,
                'images': [{'id': 1}],
            },
            {
                'annotations': [
                    {'image_id': 5, 'category_id': 3}
                ],
                'categories': ACCEPTED_CATS,
                'images': [{'id': 5}],
            }
        )
        res = divide_data(data, cat_ids, ACCEPTED_CATS)
        print(res)
        self.assertEqual(res, expected)
    def test_uneven_imgs(self):
        data = {
            'annotations': [
                {'image_id': 1, 'category_id': 1},
                {'image_id': 2, 'category_id': 3},
                {'image_id': 5, 'category_id': 3}
            ],
            'images': [
                {'id': 1},
                {'id': 2},
                {'id': 5}
            ]
        }
        cat_ids = [1, 3]
        expected = (
            {
                'annotations': [
                    {'image_id': 1, 'category_id': 1}
                ],
                'categories': ACCEPTED_CATS,
                'images': [{'id': 1}],
            },
            {
                'annotations': [
                    {'image_id': 2, 'category_id': 3},
                    {'image_id': 5, 'category_id': 3}
                ],
                'categories': ACCEPTED_CATS,
                'images': [{'id': 2}, {'id': 5}],
            }
        )
        res = divide_data(data, cat_ids, ACCEPTED_CATS)
        self.assertEqual(res, expected)

if __name__ == '__main__':
    unittest.main()
