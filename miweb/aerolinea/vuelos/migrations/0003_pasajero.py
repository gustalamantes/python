# Generated by Django 5.0.1 on 2024-01-31 15:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('vuelos', '0002_aeropuerto_alter_vuelo_destination_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Pasajero',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('first', models.CharField(max_length=64)),
                ('last', models.CharField(max_length=64)),
                ('vuelos', models.ManyToManyField(blank=True, related_name='pasajeros', to='vuelos.vuelo')),
            ],
        ),
    ]
